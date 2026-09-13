import torch

from final_project_4.src.boxes import decode_boxes
from final_project_4.src.inference import postprocess_detections


def test_decode_boxes():
    locations = torch.tensor([
        [40.0, 50.0],
        [100.0, 100.0],
    ])

    ltrb = torch.tensor([
        [
            [10.0, 20.0, 30.0, 40.0],
            [5.0, 10.0, 15.0, 20.0],
        ]
    ])

    boxes = decode_boxes(locations, ltrb)

    assert boxes.shape == (1, 2, 4)

    torch.testing.assert_close(
        boxes[0, 0],
        torch.tensor([30.0, 30.0, 70.0, 90.0]),
    )

    torch.testing.assert_close(
        boxes[0, 1],
        torch.tensor([95.0, 90.0, 115.0, 120.0]),
    )


def test_postprocess_detections_is_class_aware():
    predictions = {
        "class_logits": torch.tensor([
            [
                [10.0, -10.0],   # location 0 → class 0
                [9.0, -10.0],    # location 1 → class 0
                [-10.0, 8.0],    # location 2 → class 1
            ]
        ]),
        "box_regression": torch.tensor([
            [
                [5.0, 5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0, 5.0],
            ]
        ]),
        "centerness_logits": torch.tensor([
            [10.0, 10.0, 10.0]
        ]),
        "locations": torch.tensor([
            [20.0, 20.0],
            [21.0, 21.0],
            [20.0, 20.0],
        ]),
    }

    detections = postprocess_detections(
        predictions,
        image_size=(100, 100),
        score_threshold=0.5,
        nms_iou_threshold=0.5,
        top_k=100,
    )

    assert isinstance(detections, list)
    assert len(detections) == 1

    result = detections[0]

    # 同類別的兩個重疊框只保留高分者
    # 不同類別即使重疊仍然保留
    assert result["boxes"].shape == (2, 4)
    assert set(result["labels"].tolist()) == {0, 1}


def test_postprocess_detections_empty_result():
    predictions = {
        "class_logits": torch.full((1, 2, 3), -10.0),
        "box_regression": torch.ones((1, 2, 4)),
        "centerness_logits": torch.full((1, 2), -10.0),
        "locations": torch.tensor([
            [20.0, 20.0],
            [40.0, 40.0],
        ]),
    }

    detections = postprocess_detections(
        predictions,
        image_size=(100, 100),
        score_threshold=0.5,
    )

    result = detections[0]

    assert result["boxes"].shape == (0, 4)
    assert result["labels"].shape == (0,)
    assert result["scores"].shape == (0,)
