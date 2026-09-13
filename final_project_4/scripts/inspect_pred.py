from pathlib import Path

import torch
from torchvision.transforms import functional as TF
from torchvision.utils import draw_bounding_boxes

from final_project_4.src.data import VOC_CLASSES, build_voc_dataset
from final_project_4.src.detector import FCOSDetector
from final_project_4.src.inference import postprocess_detections
from final_project_4.src.checkpoint import load_checkpoint

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    checkpoint_path = (
        project_root / "outputs" / "checkpoints" / "pretrained_giou_longer.pt"
    )

    output_path = (
        project_root / "outputs" / "pretrained_giou_longer_prediction.jpg"
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

    model = FCOSDetector(**model_config).to(device)

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    #image_index = checkpoint["image_index"]
    #image, target = dataset[image_index]
    image, target = dataset[0]
    with torch.no_grad():
        predictions = model(image.unsqueeze(0).to(device))

        detections = postprocess_detections(
            predictions,
            image_size=tuple(image.shape[-2:]),
            score_threshold=0.05,
            nms_iou_threshold=0.5,
            top_k=20,
        )[0]

    image_uint8 = (
        image.clamp(0.0, 1.0) * 255
    ).round().to(torch.uint8)

    canvas = image_uint8

    # Ground truth：綠色
    if target["boxes"].numel() > 0:
        ground_truth_labels = [
            VOC_CLASSES[class_index.item()]
            for class_index in target["labels"]
        ]

        canvas = draw_bounding_boxes(
            canvas,
            target["boxes"],
            labels=ground_truth_labels,
            colors="green",
            width=2,
        )

    # Prediction：紅色
    if detections["boxes"].numel() > 0:
        prediction_labels = [
            f"{VOC_CLASSES[label.item()]}:{score.item():.2f}"
            for label, score in zip(
                detections["labels"],
                detections["scores"],
            )
        ]

        canvas = draw_bounding_boxes(
            canvas,
            detections["boxes"],
            labels=prediction_labels,
            colors="red",
            width=2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(canvas).save(output_path)
    
    print(
        f"checkpoint epoch: "
        f"{checkpoint['progress']['epoch']}"
    )
    print(
        f"validation loss: "
        f"{checkpoint['metrics']['validation']['loss']:.4f}"
    )
    print(f"ground-truth boxes: {target['boxes'].shape[0]}")
    print(f"predicted boxes: {detections['boxes'].shape[0]}")
    print(f"saved visualization: {output_path}")
    print("ground-truth boxes:")
    print(target["boxes"])

    print("predicted boxes:")
    print(detections["boxes"])

    print("predicted labels:")
    print(detections["labels"])

    print("predicted scores:")
    print(detections["scores"])
        

if __name__ == "__main__":
    main()
