import torch
import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from final_project_5.src.data import remap_trimap



def resize_image_and_mask(image, raw_mask, size):
    # 將 image 和 raw_mask 進行 resize，image 使用雙線性插值，raw_mask 使用最近鄰插值
    resized_image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
    resized_mask = TF.resize(raw_mask, size, interpolation=InterpolationMode.NEAREST)
    return resized_image, resized_mask

def image_and_mask_to_tensor(image, raw_mask)-> tuple[torch.Tensor, torch.Tensor]:
    # 將 image 和 raw_mask 轉換為 tensor
    image_tensor = TF.to_tensor(image)
    remapped_mask_tensor = remap_trimap(raw_mask)  # 將 raw_mask 進行 remap 也轉換為 tensor
    return image_tensor, remapped_mask_tensor

# 原始資料
# │
# ├─ image: PIL RGB
# └─ mask:  PIL L, {1,2,3}
#         │
#         ▼
#    要不要 flip？
#         │
#    ┌────┴────┐
#    │ 同一個決定 │
#    ▼         ▼
#  image     mask
#  flip      flip
#    │         │
#    └────┬────┘
#         ▼
# resize_image_and_mask()
#         │
#         ▼
# image_and_mask_to_tensors()
#         │
#         ├─ image → [3,256,256] float32
#         │
#         └─ mask  → [256,256] int64 {0,1,2}
class SegmentationTransform:
    def __init__(self, size=(256, 256), horizontal_flip_prob=0.5):
        self.size = size
        self.horizontal_flip_prob = horizontal_flip_prob

    def __call__(self, image, raw_mask):
        # ㄧ個隨機決定是否要進行水平翻轉 
        # if 0~1 取一個數 < self.horizontal_flip_prob:
        #   翻轉
        #  else:
        #   不翻轉
        should_flip = torch.rand(1).item() < self.horizontal_flip_prob
        if should_flip:
            image = TF.hflip(image)
            raw_mask = TF.hflip(raw_mask) 
        # 接著進行 resize 和轉換為 tensor   
        resized_image, resized_mask = resize_image_and_mask(image, raw_mask, self.size)
        image_tensor, mask_tensor = image_and_mask_to_tensor(resized_image, resized_mask)
        return image_tensor, mask_tensor