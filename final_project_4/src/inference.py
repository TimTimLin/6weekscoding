import torch
from torchvision.ops import batched_nms
from .boxes import clip_boxes_to_image, decode_boxes



def postprocess_detections(
    predictions: dict[str, torch.Tensor],
    image_size: tuple[int, int],
    score_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    top_k: int = 100,
) -> list[dict[str, torch.Tensor]]:
    class_probabilities = torch.sigmoid(
        predictions["class_logits"]
    )  # [B,L,C]

    centerness_probabilities = torch.sigmoid(
        predictions["centerness_logits"]
    )  # [B,L]

    scores = (
        class_probabilities
        * centerness_probabilities.unsqueeze(-1)
    )  # [B,L,C]

    boxes = decode_boxes(
        predictions["locations"],
        predictions["box_regression"],
    )  # [B,L,4]

    detections = []

    for batch_index in range(scores.shape[0]):
        image_scores = scores[batch_index]  # [L,C]

        location_indices, class_indices = torch.where(
            image_scores > score_threshold
        )

        candidate_scores = image_scores[
            location_indices,
            class_indices,
        ]

        candidate_boxes = boxes[
            batch_index,
            location_indices,
        ]

        candidate_labels = class_indices.to(torch.int64)

        if candidate_scores.numel() == 0:
            detections.append({
                "boxes": candidate_boxes.reshape(0, 4),
                "labels": candidate_labels,
                "scores": candidate_scores,
            })
            continue

        candidate_boxes = clip_boxes_to_image(
            candidate_boxes,
            image_size,
        )

        if candidate_scores.numel() > top_k:
            candidate_scores, order = candidate_scores.topk(top_k)
            candidate_boxes = candidate_boxes[order]
            candidate_labels = candidate_labels[order]

        keep = batched_nms(
            candidate_boxes,
            candidate_scores,
            candidate_labels,
            nms_iou_threshold,
        )

        detections.append({
            "boxes": candidate_boxes[keep],
            "labels": candidate_labels[keep],
            "scores": candidate_scores[keep],
        })

    return detections