from pathlib import Path
from torch.utils.data import DataLoader
from final_project_5.src.data import  build_trainval_datasets
from final_project_5.src.transforms import SegmentationTransform

train_transform = SegmentationTransform(size=(256, 256), horizontal_flip_prob=0.5)
val_transform = SegmentationTransform(size=(256, 256), horizontal_flip_prob=0.0)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

train_dataset, val_dataset = build_trainval_datasets(
    root=PROJECT_ROOT / "data",
    val_fraction=0.2,
    seed=42,
    train_transform=train_transform,
    val_transform=val_transform,
    download=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)
train_images, train_masks = next(iter(train_loader))
val_images, val_masks = next(iter(val_loader))
print(type(train_loader.sampler).__name__)
print(train_images.shape, train_images.dtype)
print(train_masks.shape, train_masks.dtype)

print(type(val_loader.sampler).__name__)
print(val_images.shape, val_images.dtype)
print(val_masks.shape, val_masks.dtype)
