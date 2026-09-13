from pathlib import Path

import torch
from torch.utils.data import DataLoader

from final_project_4.src.data import (
    VOC_CLASSES,
    build_voc_dataset,
    detection_collate_fn,
)
from final_project_4.src.detector import FCOSDetector
from final_project_4.src.evaluation import (
    evaluate_detection_model,
)
from final_project_4.src.checkpoint import load_checkpoint

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    checkpoint_path = (
        project_root
        / "outputs"
        / "checkpoints"
        / "pretrained_giou_longer.pt"
    )

    device = torch.device("cpu")

    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    model_config = checkpoint["model_config"]
    data_config = checkpoint["data"]

    dataset = build_voc_dataset(
        root=data_root,
        image_set=data_config["validation_image_set"],
        output_size=tuple(data_config["output_size"]),
        download=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )
    model = FCOSDetector(**model_config).to(device)

    # 把 model_state 裡的參數複製進 model
    model.load_state_dict(checkpoint["model_state"])
    # evaluate 模型在各類別的 AP 與整體 mAP
    metrics = evaluate_detection_model(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=model_config["num_classes"],
        score_threshold=0.05,
        nms_iou_threshold=0.5,
        map_iou_threshold=0.5,
        top_k=100,
        max_steps=100, # 固定使用前 100 個 validation batches，方便不同實驗公平比較
    )

    print(f"checkpoint epoch: {checkpoint['progress']['epoch']}")
    print(
        f"mAP@0.5 (all-point interpolation): "#把 IoU ≥ 0.5 當成成功偵測時，模型在所有類別上的平均 AP
        f"{metrics['map'].item():.4f}"
    ) 
    print()

    for class_index, class_name in enumerate(VOC_CLASSES):
        number_of_gt = int(
            metrics["num_ground_truths_per_class"][class_index].item()
        )

        if number_of_gt == 0:
            continue

        ap = metrics["ap_per_class"][class_index].item()

        print(
            f"{class_name:12s} "
            f"AP: {ap:.4f} "  #AP越高代表模型在該類別的偵測能力越好
            f"GT: {number_of_gt}"
        )


if __name__ == "__main__":
    main()
