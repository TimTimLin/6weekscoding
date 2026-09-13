import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from final_project_5.src.data import  build_trainval_datasets
from final_project_5.src.model import TinyUNet
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
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

model = TinyUNet(
    in_channels=3,
    num_classes=3,
    base_channels=32,
)
# 先取一個 batch 的資料來測試模型的 forward pass
images, masks = next(iter(val_loader))

model.eval()


with torch.inference_mode():
        logits = model(images)
        loss = F.cross_entropy(logits, masks)
        predictions = logits.argmax(dim=1)
print(images.shape, images.dtype)
print(masks.shape, masks.dtype)
print(logits.shape, logits.dtype)
print(predictions.shape, predictions.dtype)
print(torch.unique(masks))
print(torch.unique(predictions))
print(loss.item())
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
print(f"Total parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")    