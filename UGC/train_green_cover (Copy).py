import os
import csv
import random
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms as T
import pytorch_lightning as pl
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint

from torchgeo.datasets import LandCoverAI
from torchvision.transforms.functional import to_pil_image

# ---------------------------
# Config
# ---------------------------


def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


DATA_ROOT = "data/landcoverai"
BATCH_SIZE = 4


EPOCHS = 3


LR = 1e-4
NUM_WORKERS = 4
POS_CLASS = 2  # LandCover.ai class index for 'woodland' (green cover)
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
# Checkpoints are written locally so they can be synced to external storage
# (e.g., vast.ai upload or cloud bucket) when running on preemptible GPUs.
CHECKPOINT_EVERY_N_EPOCHS = _int_env("CHECKPOINT_EVERY_N_EPOCHS", 1)
CHECKPOINT_EVERY_N_STEPS = _int_env("CHECKPOINT_EVERY_N_STEPS", 0)
METRICS_PATH = OUT_DIR / "checkpoint_metrics.csv"
RESUME_FROM_LAST = os.environ.get("RESUME_FROM_LAST", "1").lower() not in {"0", "false", "no", "off"}
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

# ---------------------------
# Dataset wrapper: convert multiclass mask -> binary (green vs non-green)
# ---------------------------
class BinaryGreenCover(torch.utils.data.Dataset):
    def __init__(self, base_ds, size=512, train=True):
        self.base_ds = base_ds
        self.train = train

        self.to_tensor = T.Compose([
            T.ToTensor(),  # [0,1]
        ])

        self.crop = T.RandomCrop(size) if train else T.CenterCrop(size)

        # Normalization (ImageNet-ish for torchvision backbones)
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]  # dict with 'image' (PIL) and 'mask' (PIL or np)
        img = sample["image"]
        mask = sample["mask"]  # HxW with class indices {0..4}

        # ensure PIL inputs for torchvision spatial transforms
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if isinstance(mask, np.ndarray):
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask)
        if not isinstance(mask, Image.Image):
            mask = to_pil_image(mask)

        # crop first (same crop for img and mask)
        crop_size = self.crop.size
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        if self.train:
            i, j, h, w = T.RandomCrop.get_params(img, output_size=crop_size)
            img = T.functional.crop(img, i, j, h, w)
            mask = T.functional.crop(mask, i, j, h, w)
        else:
            img = T.functional.center_crop(img, crop_size)
            mask = T.functional.center_crop(mask, crop_size)

        # binary mask: 1 where woodland, else 0
        mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)
        bin_mask = (mask_tensor == POS_CLASS).float().unsqueeze(0)  # (1,H,W)

        # to tensor + normalize
        img = self.to_tensor(img)  # (3,H,W)

        if self.train:
            if random.random() < 0.5:
                img = T.functional.hflip(img)
                bin_mask = torch.flip(bin_mask, dims=[2])
            if random.random() < 0.5:
                img = T.functional.vflip(img)
                bin_mask = torch.flip(bin_mask, dims=[1])

        img = self.norm(img)

        return img, bin_mask


# ---------------------------
# LightningModule for simple training loop
# ---------------------------
class GreenSegModel(pl.LightningModule):
    def __init__(self, lr=LR):
        super().__init__()
        self.save_hyperparameters()
        self.model = deeplabv3_resnet50(weights=None, num_classes=1)  # binary logits
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.f1 = BinaryF1Score(threshold=0.5)
        self.iou = BinaryJaccardIndex(threshold=0.5)
        self.lr = lr

    def forward(self, x):
        return self.model(x)["out"]  # (B,1,H,W)

    def _shared_step(self, batch, stage="train"):
        x, y = batch  # y in {0,1}, shape (B,1,H,W)
        logits = self(x)
        loss = self.loss_bce(logits, y)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_f1", self.f1(probs, y.int()), prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}_iou", self.iou(probs, y.int()), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")

    def test_step(self, batch, _):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)


class CheckpointMetricsLogger(pl.Callback):
    """Append key metrics to a CSV so they can be inspected or synced with checkpoints."""

    def __init__(self, path):
        self.path = Path(path)
        self.fieldnames = ["timestamp", "epoch", "global_step", "val_loss", "val_f1", "val_iou", "train_loss"]
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    @staticmethod
    def _to_scalar(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().mean().item()
        if isinstance(value, (float, int)):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        if train_loss is None:
            train_loss = metrics.get("train_loss_epoch")
        row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "val_loss": self._to_scalar(metrics.get("val_loss")),
            "val_f1": self._to_scalar(metrics.get("val_f1")),
            "val_iou": self._to_scalar(metrics.get("val_iou")),
            "train_loss": self._to_scalar(train_loss),
        }
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def get_dataloaders():
    # TorchGeo dataset (downloads automatically on first use)
    base = LandCoverAI(root=DATA_ROOT, split="train", download=True)  # images + masks

    # quick train/val split
    n = len(base)
    n_train = int(0.8 * n)
    n_val = n - n_train
    base_train, base_val = random_split(base, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_ds = BinaryGreenCover(base_train, size=512, train=True)
    val_ds = BinaryGreenCover(base_val, size=512, train=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_dl, val_dl


def save_visual_example(model, ds, fname="preview.png"):
    model.eval()
    device = next(model.parameters()).device
    x, y = ds[0]  # one sample
    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits)[0, 0].cpu()
        pred = (prob > 0.5).float()

    # stack preview: RGB image + GT mask + Pred mask
    # Note: input already normalized; undo for nicer preview
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    x_img = torch.clamp(x * std + mean, 0, 1)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1); plt.title("Image"); plt.axis("off"); plt.imshow(x_img.permute(1,2,0))
    plt.subplot(1,3,2); plt.title("GT green"); plt.axis("off"); plt.imshow(y[0], cmap="gray")
    plt.subplot(1,3,3); plt.title("Pred green"); plt.axis("off"); plt.imshow(pred, cmap="gray")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.close()


def main():
    print(f"Using device: {DEVICE}")
    train_dl, val_dl = get_dataloaders()

    model = GreenSegModel(lr=LR)
    model = model.to(DEVICE)

    # Save checkpoints on a configurable schedule so interrupted runs (e.g., on preemptible cloud GPUs) can resume.
    every_n_epochs = CHECKPOINT_EVERY_N_EPOCHS if CHECKPOINT_EVERY_N_EPOCHS > 0 else None
    every_n_steps = CHECKPOINT_EVERY_N_STEPS if CHECKPOINT_EVERY_N_STEPS > 0 else None
    if every_n_epochs is None and every_n_steps is None:
        every_n_epochs = 1  # default safety net
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="green-cover-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=every_n_epochs,
        every_n_train_steps=every_n_steps,
    )
    metrics_logger = CheckpointMetricsLogger(METRICS_PATH)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, metrics_logger],
    )
    ckpt_path = None
    if RESUME_FROM_LAST:
        last_ckpt = CHECKPOINT_DIR / "last.ckpt"
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)
            print(f"Resuming from checkpoint: {ckpt_path}")
        else:
            print("No checkpoint found; starting fresh.")
    else:
        print("RESUME_FROM_LAST is disabled; starting fresh and writing new checkpoints.")

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt_path)

    # ensure model stays on the training device for downstream usage
    model = model.to(DEVICE)

    # quick preview
    val_ds = val_dl.dataset
    save_visual_example(model, val_ds, fname="preview.png")

    # export a TorchScript for simple inference (optional)
    model.eval()
    example = next(iter(val_dl))[0].to(next(model.parameters()).device)
    traced = torch.jit.trace(lambda x: model(x), example)
    torch.jit.save(traced, OUT_DIR / "green_seg_deeplabv3.pt")
    print(f"Saved preview and model to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
