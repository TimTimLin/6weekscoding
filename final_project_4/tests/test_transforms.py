import torch
import pytest

from final_project_4.src.boxes import clip_boxes_to_image, box_area
from final_project_4.src.transforms import resize_image_and_boxes

def test_clip_boxes_to_image_and_preserve_input():
    boxes = torch.tensor([
    [-10.0, 20.0, 170.0, 120.0],
    [0.0, 0.0, 160.0, 100.0],
    ])
    original_boxes = boxes.clone()
    image_size = (100, 160)
    clipped_boxes = clip_boxes_to_image(boxes, image_size)

    expected_clipped_boxes = torch.tensor([
        [0.0, 20.0, 160.0, 100.0],      
        [0.0, 0.0, 160.0, 100.0],
    ])
   

    assert clipped_boxes.shape == expected_clipped_boxes.shape
    torch.testing.assert_close(clipped_boxes, expected_clipped_boxes)
    torch.testing.assert_close(boxes, original_boxes)

def test_full_image_box_uses_continuous_boundary():
    boxes = torch.tensor([
        [-10.0, -10.0, 170.0, 120.0],
    ])
    expected = torch.tensor([
        [0.0, 0.0, 160.0, 100.0],
    ])
    original_boxes = boxes.clone()  # Keep a copy of the original boxes
    clipped_boxes = clip_boxes_to_image(boxes, (100, 160))
    torch.testing.assert_close(clipped_boxes, expected)

    areas = box_area(clipped_boxes)
    expected_areas = torch.tensor([16000.0])
    torch.testing.assert_close(boxes, original_boxes)
    torch.testing.assert_close(areas, expected_areas)

    

def test_resize_image_and_boxes():
    image = torch.rand(3, 120, 200, dtype=torch.float32)     # Random image with shape (C, H, W)
    boxes = torch.tensor([
        [20.0, 30.0, 100.0, 90.0],
    ])
    original_image = image.clone()  # Keep a copy of the original image
    original_boxes = boxes.clone()  # Keep a copy of the original boxes
    resized_image, resized_boxes = resize_image_and_boxes(
    image,
    boxes,
    output_size=(300, 400),
    )
    
    expected_resized_boxes = torch.tensor([[40.0, 75.0, 200.0, 225.0]])
    assert resized_image.shape == torch.Size([3, 300, 400])  
    assert resized_boxes.shape == torch.Size([1, 4])  
    torch.testing.assert_close(resized_boxes, expected_resized_boxes)
    assert resized_image.dtype == image.dtype
    assert resized_boxes.dtype == boxes.dtype
    torch.testing.assert_close(boxes, original_boxes)
    torch.testing.assert_close(image, original_image)