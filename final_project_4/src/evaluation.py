import torch

from final_project_4.src.inference import postprocess_detections
from final_project_4.src.metrics import (
    compute_mean_average_precision,
)

def evaluate_detection_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_classes: int,
    score_threshold: float=0.05,
    nms_iou_threshold: float=0.5, #  prediction 和 prediction 之間 同類別prediction 彼此的IOU都過高 只保留分數最高的prediction
    map_iou_threshold: float = 0.5,#  prediction 和 GT 之間 透過 IoU 來決定是否為 TP 或 FP
    top_k: int=100,
    max_steps: int|None=None,
    
)->dict[str, torch.Tensor]:
    model.eval()

    all_detections = []    
    all_targets = []
    num_batches = 0
    # 只做預測不須要計算梯度，使用 torch.inference_mode() 可以節省記憶體和加速推論
    with torch.inference_mode():
        # 每一張圖片都會經過 dataloader 進行預測
        for step, (images, targets) in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
            
            images = images.to(device)
            # 得到模型的初始輸出結果(raw_predictions)，包含了每個位置的預測框、類別分數、中心度分數等資訊
            raw_predictions = model(images)
            # 進行後處理，得到每張圖片prediction的boxes、labels、scores
            detections = postprocess_detections(
                predictions=raw_predictions,
                image_size=tuple(images.shape[-2:]),
                score_threshold=score_threshold,
                nms_iou_threshold=nms_iou_threshold,
                top_k=top_k,
            )
            # 將每張圖片的預測結果和真實標籤分別存入 all_detections 和 all_targets
            for detection, target in zip(detections, targets):
                all_detections.append({
                    "boxes": detection["boxes"].detach().to(torch.float32).cpu(),
                    "labels": detection["labels"].detach().to(torch.int64).cpu(),
                    "scores": detection["scores"].detach().to(torch.float32).cpu(),
                })
                all_targets.append({
                    "boxes": target["boxes"].to(torch.float32).cpu(),
                    "labels": target["labels"].to(torch.int64).cpu(),
                    "difficult": target["difficult"].to(torch.bool).cpu(),
                })  
            num_batches += 1
        if num_batches == 0:
            raise ValueError("No batches were processed. Please check the dataloader and max_steps parameter.")
    # 計算 mAP 流程已包含match_single_class_predictions()、計算每個 class 的 AP，並將所有 class 的 AP 平均得到 mAP
    metrics = compute_mean_average_precision(
        predictions=all_detections,
        targets=all_targets,
        num_classes=num_classes,
        iou_threshold=map_iou_threshold,#
    )
    return metrics