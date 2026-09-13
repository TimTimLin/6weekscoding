from torch import nn
import torch

from .model import (
    ResNet18Backbone,
    FeaturePyramidNetwork,
    DetectionHead,
)
from .targets import flatten_predictions, generate_locations



class FCOSDetector(nn.Module):
    def __init__(self, num_classes: int = 20, fpn_channels: int = 128, pretrained: bool = False, normalize_inputs: bool = False) -> None:
        super().__init__()

        self.backbone = ResNet18Backbone(pretrained=pretrained)
        self.fpn = FeaturePyramidNetwork(out_channels=fpn_channels)
        self.head = DetectionHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
        )

        self.strides = {
            "p3": 8,
            "p4": 16,
            "p5": 32,
        }
        
        self.normalize_inputs = normalize_inputs
        # 模型要保存、要跟著 device 移動，但不要學習的 Tensor
        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )
    # 定義一個方法來正規化輸入的影像，將影像的像素值轉換為標準化的形式
    def normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        if not self.normalize_inputs:
            return images

        mean = self.image_mean.to(dtype=images.dtype)
        std = self.image_std.to(dtype=images.dtype)

        return (images - mean) / std    
        
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        #先將輸入的影像進行正規化處理，將影像的像素值轉換為標準化的形式
        images = self.normalize_images(images)
        #從images中提取特徵，通過backbone提取特徵
        features = self.backbone(images)
        #通過FPN將不同層(p3, p4, p5)的特徵融合，得到多尺度的特徵圖
        pyramid = self.fpn(features)
        #通過head對融合後的特徵圖進行分類和回歸，得到每個位置的class_logits、box_regression和centerness_logits
        level_predictions = self.head(pyramid)
        #將多層級的預測結果展平，方便後續計算損失
        flattened_predictions = flatten_predictions(level_predictions)
        #生成每個位置的預測的實際座標和對應的stride，方便後續計算損失
        locations, strides = generate_locations(level_predictions, self.strides)

        return {
                "class_logits": flattened_predictions["class_logits"],   #[B,L,20]
                "box_regression": flattened_predictions["box_regression"],#[B,L,4]
                "centerness_logits": flattened_predictions["centerness_logits"], #[B,L]
                "locations": locations, #[L,2]
                "strides": strides #[L]
            }