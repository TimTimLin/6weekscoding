import torch

from final_project_4.src.losses import compute_detection_loss
from final_project_4.src.targets import assign_targets_to_locations

# Dataset GT
# image 1:
#   box1 = [40,40,120,120], label=3
#
#        ↓
# build_batch_targets()
#        ↓
# 對每張圖片執行 assign_targets_to_locations()
#        ↓
# 將 GT 對應到每個 location，決定每個 location 要負責哪個 GT
#
# image 1:
#   location 0 → label=3, box_target, centerness
#   location 1 → background
#   location 2 → background
#   location 3 → label=3, box_target, centerness
#   location 4 → label=3, box_target, centerness
#   ...
def build_batch_targets(
    locations: torch.Tensor,  #[L, 2]
    locations_strides: torch.Tensor, #[L]
    dataset_targets: list[dict[str, torch.Tensor]],  #每張圖片的targets
    regression_ranges: dict[int, tuple[float, float]],
) -> list[dict[str, torch.Tensor]]:
    #對每張圖片的targets進行處理，將每個位置對應的正確答案邊界框和正確標籤分配給每個位置
    batch_targets = []
    for target_image in dataset_targets:
        #將每張圖片的targets中的boxes和labels提取出來，並移動到與locations相同的裝置和數據類型上
        boxes = target_image["boxes"].to(locations.device,dtype=locations.dtype)
        labels = target_image["labels"].to(locations.device,dtype=torch.int64)
        #將每張圖片的targets中的boxes和labels分配給每個位置，得到每個位置對應的正確答案邊界框和正確標籤
        image_targets = assign_targets_to_locations(
            locations=locations,
            locations_strides=locations_strides,
            boxes=boxes,
            labels=labels,
            regression_ranges=regression_ranges,
        )
        batch_targets.append(image_targets)
    return batch_targets
# Dataset
#    ↓
# DataLoader
#    ↓
# ┌────────────────────────────┐
# │      train_one_epoch()     │
# │                            │
# │ Batch 1 → train_one_step() │
# │ Batch 2 → train_one_step() │
# │ Batch 3 → train_one_step() │
# │ ...                        │
# │ Batch N → train_one_step() │
# │                            │
# │ 累積每個 batch 的 loss      │
# │ 最後計算平均               │
# └────────────────────────────┘
#    ↓
# 這一個 epoch 的平均 loss
def train_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    dataset_targets: list[dict[str, torch.Tensor]],
    regression_ranges: dict[int, tuple[float, float]],
    num_classes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> dict[str, torch.Tensor]:
    #將模型設置為訓練模式
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(images)
    #將每張圖片的targets中的boxes和labels分配給每個位置，得到每個位置對應的正確答案邊界框和正確標籤
    assigned_targets = build_batch_targets(
        predictions["locations"],
        predictions["strides"],
        dataset_targets,
        regression_ranges,
        )
    #計算predictions和assigned_targets之間的損失，包括分類損失、邊界框回歸損失和中心度損失
    losses = compute_detection_loss(
        predictions,
        assigned_targets,
        num_classes,
        alpha,
        gamma
    )
    #進行反向傳播，計算梯度
    losses["loss"].backward()
    #更新模型參數
    optimizer.step()
    return {
        name: loss.detach() for name, loss in losses.items()
    }
    
def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    regression_ranges: dict[int, tuple[float, float]],
    num_classes: int,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.train()
    total_losses = {
        "classification_loss": 0.0,
        "box_regression_loss": 0.0,
        "centerness_loss": 0.0,
        "loss": 0.0,
    }
    num_batches = 0
    #從dataloader中獲取每個batch的images和dataset_targets來train
    for step, (images, dataset_targets) in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        
        images = images.to(device)
        
        losses = train_one_step(
            model=model,
            optimizer=optimizer,
            images=images,
            dataset_targets=dataset_targets,
            regression_ranges=regression_ranges,
            num_classes=num_classes,
        )
        for name in total_losses:
            total_losses[name] += losses[name].item()
            
        num_batches += 1
    if num_batches == 0:
        raise ValueError("No batches were processed. Check your dataloader.")
    
    return {
        name: total_loss / num_batches for name, total_loss in total_losses.items()
    }
    
def evaluate_one_epoch(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    regression_ranges: dict[int, tuple[float, float]],
    num_classes: int,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_losses = {
        "classification_loss": 0.0,
        "box_regression_loss": 0.0,
        "centerness_loss": 0.0,
        "loss": 0.0,
    }
    num_batches = 0
    with torch.no_grad():
        for step, (images, dataset_targets) in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
            
            images = images.to(device)
            
            predictions = model(images)
            assigned_targets = build_batch_targets(
                predictions["locations"],
                predictions["strides"],
                dataset_targets,
                regression_ranges,
            )
            losses = compute_detection_loss(
                predictions,
                assigned_targets,
                num_classes,
            )
            for name in total_losses:
                total_losses[name] += losses[name].item()
                
            num_batches += 1
            
    if num_batches == 0:    
            raise ValueError("No batches were processed. Check your dataloader.")
    return {
                name: total_loss / num_batches for name, total_loss in total_losses.items()
            }