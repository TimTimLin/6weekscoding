

import torch
from final_project_5.src.metrics import (
    confusion_matrix_from_predictions,
    iou_from_confusion_matrix,
)

def train_step(
    model,
    images: torch.Tensor,
    masks: torch.Tensor,
    optimizer,
    loss_fn,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = loss_fn(logits, masks)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device,
) -> float:
    model.train()
    total_loss = 0.0
    # 有幾個樣本
    num_samples = 0
    # 把dataloader裡的每個batch都拿出來做train_step
    for images, masks in dataloader:
        
        images = images.to(device)
        masks = masks.to(device)
        batch_loss = train_step(model, images, masks, optimizer, loss_fn)
        batch_size = images.size(0)
        total_loss += batch_loss * batch_size
        num_samples += batch_size
        
    if num_samples == 0:
            raise ValueError(
                "Cannot train on an empty dataloader."
            )

    return total_loss / num_samples

def evaluate(
    model,
    dataloader,
    loss_fn,
    device,
    num_classes: int,
) -> dict:
    model.eval()
    with torch.inference_mode():
        total_loss = 0.0
        num_samples = 0
        total_confusion_matrix = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.int64,
            device=device,
        )
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = loss_fn(logits, masks)
            predictions = torch.argmax(logits, dim=1)
            batch_confusion_matrix = confusion_matrix_from_predictions(
                predictions,
                masks,
                num_classes=num_classes,
            )
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            
            total_confusion_matrix += batch_confusion_matrix
            
    if num_samples == 0 or total_confusion_matrix.sum().item() == 0:
            raise ValueError(
                "Cannot evaluate on an empty dataloader."
            )
    average_loss = total_loss / num_samples
    per_class_iou, mean_iou = iou_from_confusion_matrix(total_confusion_matrix)
    return {
        "loss": average_loss,
        "per_class_iou": per_class_iou.cpu(),
        "mean_iou": mean_iou.item(),
        "confusion_matrix": total_confusion_matrix.cpu(),
    }
        
            
    

