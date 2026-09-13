
import torch

def box_area(boxes: torch.Tensor) -> torch.Tensor:
    #(x_min, y_min, x_max, y_max)
    #x_max - x_min = width
    #y_max - y_min = height
    width = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    height = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    area = width * height
    return area

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    #計算左上和右下角的座標
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N,M,2]
    #計算交集的寬度和高度，避免出現負值
    wh = (rb - lt).clamp(min=0)  # [N,M,2] (width, height)
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N,M]  (width * height)
    #計算聯集的面積，避免除以零
    union = (area1[:, None] + area2[None, :] - intersection).clamp(min=1e-7)  
    
    iou = intersection / union
    return iou

def clip_boxes_to_image(boxes: torch.Tensor,image_size: tuple[int, int]) -> torch.Tensor:
    clipped_boxes = boxes.clone()
    h, w = image_size
    #將邊界框的座標限制在圖像範圍內
    clipped_boxes[:, 0] = boxes[:, 0].clamp(min=0, max=w)
    clipped_boxes[:, 1] = boxes[:, 1].clamp(min=0, max=h)
    clipped_boxes[:, 2] = boxes[:, 2].clamp(min=0, max=w)
    clipped_boxes[:, 3] = boxes[:, 3].clamp(min=0, max=h)
    return clipped_boxes


def decode_boxes(
    locations: torch.Tensor,  # [L,2]
    ltrb: torch.Tensor,       # [B,L,4]
) -> torch.Tensor:            # [B,L,4]
    x = locations[:, 0].unsqueeze(0)  # [1,L]
    y = locations[:, 1].unsqueeze(0)  # [1,L]

    l = ltrb[..., 0]  # [B,L]
    t = ltrb[..., 1]
    r = ltrb[..., 2]
    b = ltrb[..., 3]

    return torch.stack(
        [
            x - l,
            y - t,
            x + r,
            y + b,
        ],
        dim=-1,
    )
