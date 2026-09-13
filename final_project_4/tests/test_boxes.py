import torch
import pytest

from final_project_4.src.boxes import box_area, box_iou

def test_box_area():
    boxes = torch.tensor([
        [0.0, 0.0, 2.0, 3.0],
        [1.0, 1.0, 4.0, 5.0]
    ])
    expected_areas = torch.tensor([6.0, 12.0])
    areas = box_area(boxes)
    assert areas.shape == expected_areas.shape
    torch.testing.assert_close(areas, expected_areas)
    
def test_box_iou():
    boxes1 = torch.tensor([
    [0.0, 0.0, 2.0, 2.0],
    [0.0, 0.0, 1.0, 1.0],
    ])

    boxes2 = torch.tensor([
        [0.0, 0.0, 2.0, 2.0],
        [1.0, 1.0, 3.0, 3.0],
    ])
    expected_iou = torch.tensor([[1.0, 1/7], [0.25, 0.0]])
    iou = box_iou(boxes1, boxes2)
    assert iou.shape == expected_iou.shape
    # Ensure that all values in the IoU tensor are finite (not NaN or Inf)
    assert torch.isfinite(iou).all().item()
    torch.testing.assert_close(iou, expected_iou)

def test_box_iou_no_intersection():
    boxes1 = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],
    ])

    boxes2 = torch.tensor([
        [2.0, 2.0, 3.0, 3.0],
    ])
    expected_iou = torch.tensor([[0.0]])
    iou = box_iou(boxes1, boxes2)
    assert iou.shape == expected_iou.shape
    torch.testing.assert_close(iou, expected_iou)

def test_box_iou_zero_area():
    boxes1 = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],  # Zero width
    ])

    boxes2 = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],
    ])
    expected_iou = torch.tensor([[0.0]])
    iou = box_iou(boxes1, boxes2)
    assert iou.shape == expected_iou.shape
    torch.testing.assert_close(iou, expected_iou)