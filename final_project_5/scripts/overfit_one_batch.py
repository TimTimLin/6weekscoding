

import torch
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from final_project_5.src.data import build_trainval_datasets
from final_project_5.src.engine import train_step
from final_project_5.src.model import TinyUNet
from final_project_5.src.transforms import SegmentationTransform
from final_project_5.src.metrics import (
    confusion_matrix_from_predictions,
    iou_from_confusion_matrix,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_transform = SegmentationTransform(
    size=(256, 256),
    horizontal_flip_prob=0.0,
)

val_transform = SegmentationTransform(
    size=(256, 256),
    horizontal_flip_prob=0.0,
)

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

images, masks = next(iter(train_loader))
images = images.to(device)
masks = masks.to(device)
model = TinyUNet(
    in_channels=3,
    num_classes=3,
    base_channels=32,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

loss_fn = torch.nn.CrossEntropyLoss()
num_steps = 500

loss_history = []

for step in range(1, num_steps + 1):
    train_step_loss = train_step(
        model=model,
        images=images,
        masks=masks,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )
    loss_history.append(train_step_loss)
    if step == 1 or step % 25 == 0:
        print(f"Step {step}: Loss = {train_step_loss:.6f}")

print(f"Initial loss: {loss_history[0]:.6f}")
print(f"Final loss:   {loss_history[-1]:.6f}")

model.eval()

with torch.inference_mode():
    logits = model(images)
    predictions = logits.argmax(dim=1)

    confusion_matrix = confusion_matrix_from_predictions(
        predictions,
        masks,
        num_classes=3,
    )

    per_class_iou, mean_iou = iou_from_confusion_matrix(
        confusion_matrix
    )

class_counts = torch.bincount(
    masks.reshape(-1),
    minlength=3,
)

class_ratios = (
    class_counts.float()
    / class_counts.sum()
)

print("Class counts:", class_counts.cpu().tolist())
print("Class ratios:", class_ratios.cpu().tolist())
print("Confusion matrix:")
print(confusion_matrix.cpu())
print("Per-class IoU:", per_class_iou.cpu().tolist())
print("Mean IoU:", mean_iou.item())