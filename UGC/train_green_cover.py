import os
import csv
from pathlib import Path
from datetime import datetime
import torch
from lightning.pytorch import Trainer, Callback
from lightning.pytorch.callbacks import ModelCheckpoint
import kornia.augmentation as K

from torchgeo.datasets import LandCoverAI
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.trainers import SemanticSegmentationTask

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


EPOCHS = 15


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
CROP_SIZE = _int_env("CROP_SIZE", 512)
METRICS_PATH = OUT_DIR / "checkpoint_metrics.csv"
RESUME_FROM_LAST = os.environ.get("RESUME_FROM_LAST", "1").lower() not in {"0", "false", "no", "off"}
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

# ---------------------------
# TorchGeo dataset + datamodule wrappers
# ---------------------------
class LandCoverAIGreenCover(LandCoverAI):
    """LandCover.ai sample with woodland class collapsed to a binary mask."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        image = sample["image"].float() / 255.0  # scale to [0,1]
        mask = (sample["mask"] == POS_CLASS).long()
        sample["image"] = image
        sample["mask"] = mask
        return sample


class BinaryLandCoverAIDataModule(NonGeoDataModule):
    """DataModule that applies TorchGeo augmentations and binary mask handling."""

    def __init__(
        self,
        root: str,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        crop_size: int | None = 512,
        download: bool = True,
    ) -> None:
        super().__init__(
            LandCoverAIGreenCover,
            batch_size=batch_size,
            num_workers=num_workers,
            root=root,
            download=download,
        )
        self.crop_size = crop_size
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        aug_ops = []
        if crop_size is not None and crop_size > 0:
            aug_ops.append(K.RandomCrop((crop_size, crop_size)))
        aug_ops.extend(
            [
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomRotation(p=0.5, degrees=90),
                K.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.train_aug = K.AugmentationSequential(*aug_ops, data_keys=None, keepdim=True)
        self.aug = K.AugmentationSequential(K.Normalize(mean=self.mean, std=self.std), data_keys=None, keepdim=True)

class CheckpointMetricsLogger(Callback):
    """Append key metrics to a CSV so they can be inspected or synced with checkpoints."""

    def __init__(self, path):
        self.path = Path(path)
        self.fieldnames = [
            "timestamp",
            "epoch",
            "global_step",
            "val_loss",
            "val_accuracy",
            "val_iou",
            "train_loss",
        ]
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
            "val_accuracy": self._to_scalar(metrics.get("val_Accuracy")),
            "val_iou": self._to_scalar(metrics.get("val_JaccardIndex")),
            "train_loss": self._to_scalar(train_loss),
        }
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_visual_example(task: SemanticSegmentationTask, datamodule: BinaryLandCoverAIDataModule, fname="preview.png"):
    task.eval()
    device = next(task.parameters()).device
    datamodule.setup("fit")
    val_dataset = datamodule.val_dataset
    if val_dataset is None:
        datamodule.setup("validate")
        val_dataset = datamodule.val_dataset
    assert val_dataset is not None, "Validation dataset is not available."
    sample = val_dataset[0]
    image = sample["image"]
    mask = sample["mask"]
    mean = datamodule.mean.view(3, 1, 1)
    std = datamodule.std.view(3, 1, 1)
    image_norm = (image - mean) / std
    with torch.no_grad():
        logits = task(image_norm.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits)[0, 0].cpu()
        pred = (prob >= 0.5).float()

    # stack preview: RGB image + GT mask + Pred mask
    x_img = torch.clamp(image, 0, 1)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1); plt.title("Image"); plt.axis("off"); plt.imshow(x_img.permute(1,2,0))
    plt.subplot(1,3,2); plt.title("GT green"); plt.axis("off"); plt.imshow(mask.cpu(), cmap="gray")
    plt.subplot(1,3,3); plt.title("Pred green"); plt.axis("off"); plt.imshow(pred, cmap="gray")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.close()


def main():
    print(f"Using device: {DEVICE}")
    datamodule = BinaryLandCoverAIDataModule(
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        crop_size=CROP_SIZE,
        download=True,
    )

    task_kwargs = dict(
        model='deeplabv3+',
        backbone='resnet50',
        weights=True,
        in_channels=3,
        task='binary',
        loss='bce',
        lr=LR,
        patience=5,
    )

    task: SemanticSegmentationTask | None = None
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

    trainer = Trainer(
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
            # Check compatibility of stored state dict with current TorchGeo task.
            ckpt_state = torch.load(last_ckpt, map_location="cpu")
            state_dict = ckpt_state.get("state_dict", {})
            incompatible = any(key.startswith("model.backbone") for key in state_dict.keys())
            del ckpt_state
            if incompatible:
                print(
                    "Found checkpoint from a previous script version; skipping automatic resume. "
                    "Set RESUME_FROM_LAST=0 or remove outputs/checkpoints if you want a clean start."
                )
            else:
                ckpt_path = str(last_ckpt)
                print(f"Resuming from checkpoint: {ckpt_path}")
        else:
            print("No checkpoint found; starting fresh.")
    else:
        print("RESUME_FROM_LAST is disabled; starting fresh and writing new checkpoints.")

    if task is None:
        task = SemanticSegmentationTask(**task_kwargs)

    trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)

    # ensure model stays on the training device for downstream usage
    task = task.to(DEVICE)

    # quick preview
    save_visual_example(task, datamodule, fname="preview.png")

    # export a TorchScript for simple inference (optional)
    task.eval()
    datamodule.setup("fit")
    val_dataset = datamodule.val_dataset
    assert val_dataset is not None, "Validation dataset is required for exporting."
    sample = val_dataset[0]
    mean = datamodule.mean.view(3, 1, 1)
    std = datamodule.std.view(3, 1, 1)
    example = ((sample["image"] - mean) / std).unsqueeze(0).to(next(task.parameters()).device)
    traced = torch.jit.trace(lambda x: task.model(x), example)
    torch.jit.save(traced, OUT_DIR / "green_seg_deeplabv3.pt")
    print(f"Saved preview and model to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
