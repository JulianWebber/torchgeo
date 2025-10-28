import os
import random
from pathlib import Path
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

from torchgeo.datasets import LandCoverAI
from torchvision.transforms.functional import to_pil_image

# ---------------------------
# Config
# ---------------------------
DATA_ROOT = "data/landcoverai"
BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 4
POS_CLASS = 2  # LandCover.ai class index for 'woodland' (green cover)
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    x, y = ds[0]  # one sample
    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(DEVICE))
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

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        enable_checkpointing=False
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # quick preview
    val_ds = val_dl.dataset
    save_visual_example(model, val_ds, fname="preview.png")

    # export a TorchScript for simple inference (optional)
    model.eval()
    example = next(iter(val_dl))[0].to(DEVICE)
    traced = torch.jit.trace(lambda x: model(x), example)
    torch.jit.save(traced, OUT_DIR / "green_seg_deeplabv3.pt")
    print(f"Saved preview and model to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
