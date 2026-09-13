import torch
from torchvision.transforms import functional as TF

from final_project_4.src.boxes import clip_boxes_to_image

def resize_image_and_boxes(image:torch.Tensor, boxes:torch.Tensor, output_size:tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    old_height, old_width = image.shape[-2:]
    new_height, new_width = output_size
    sx = new_width / old_width
    sy = new_height / old_height
    # Resize the image
    resized_image = TF.resize(image, [new_height, new_width], antialias=True)

    # Adjust the bounding boxes
    resized_boxes = boxes.clone()
    resized_boxes[:, [0, 2]] *= sx  # Scale x_min and x_max
    resized_boxes[:, [1, 3]] *= sy  # Scale y_min and y_max

    # Clip the boxes to ensure they are within the image boundaries
    resized_boxes = clip_boxes_to_image(resized_boxes, output_size)

    return resized_image, resized_boxes