import torch
from torch import nn
import math
from torchvision.models import ResNet18_Weights, resnet18
from torch.nn import functional as F
class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained:bool=False)-> None:
        super().__init__()
        # 如果pretrained=True，則使用ResNet18_Weights.DEFAULT權重初始化模型，否則不使用預訓練權重
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        # 直接使用resnet18函數建立ResNet18模型，就不用自己定義ResNet18的函式 
        #  ex nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
        #  ex nn.BatchNorm2d(in_channels),
        backbone = resnet18(weights=weights)
        # self.stem放著預計要使用的ResNet18的前幾層，包含conv1, bn1, relu, maxpool
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool 
        )
        #backbone.layer1
        # │
        # ├── BasicBlock
        # │   ├── Conv2d
        # │   ├── BatchNorm
        # │   ├── ReLU
        # │   ├── Conv2d
        # │   ├── BatchNorm
        # │   └── shortcut + ReLU
        # │
        # └── BasicBlock
        #     ├── Conv2d
        #     ├── BatchNorm
        #     ├── ReLU
        #     ├── Conv2d
        #     ├── BatchNorm
        #     └── shortcut + ReLU
        # 一個layer包含兩個BasicBlock
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        # 不使用avgpool和fc，因為我們只需要提取特徵圖，不用分類 
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]: #input: images[B, 3, H, W]

        #stem(images)
        # → conv1(images)
        # → bn1(...)
        # → relu(...)
        # → maxpool(...)
        x = self.stem(images)  #stride=4, output: [B, 64, H/4, W/4]
        
        c2 = self.layer1(x) #stride=4, output: [B, 64, H/4, W/4]
        c3 = self.layer2(c2) #stride=8, output: [B, 128, H/8, W/8]
        c4 = self.layer3(c3) #stride=16, output: [B, 256, H/16, W/16]
        c5 = self.layer4(c4) ##stride=32, output: [B, 512, H/32, W/32]
        return { "c3": c3, "c4": c4, "c5": c5 }

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, out_channels: int=128) -> None:
        super().__init__()
        # 1x1卷積層，將輸入特徵圖的通道數轉換為out_channels
        self.lateral_c3 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(512, out_channels, kernel_size=1)
        # 3x3卷積層，對融合後的特徵圖進行進一步處理
        self.output_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        c3, c4, c5 = features["c3"], features["c4"], features["c5"]
        m5 = self.lateral_c5(c5) 
        
        #合併c4和m5，將m5上採樣到c4的大小，然後相加
        m4 = self.lateral_c4(c4) + F.interpolate(
            m5,
            size=c4.shape[-2:], #做upsample，將m5的高寬調整成c4的高寬，不乘2是怕有誤差，直接用c4的高寬
            mode="nearest",
        )
        #合併c3和m4，將m4上採樣到c3的大小，然後相加
        m3 = self.lateral_c3(c3) + F.interpolate(
            m4,
            size=c3.shape[-2:], #做upsample，將m4的高寬調整成c3的高寬，不乘2是怕有誤差，直接用c3的高寬
            mode="nearest",
        )
        p3 = self.output_p3(m3)
        p4 = self.output_p4(m4)
        p5 = self.output_p5(m5)

        return {"p3": p3, "p4": p4, "p5": p5}


# P3/P4/P5
#     ├── classification tower
#     │       3×3 Conv → ReLU
#     │       3×3 Conv → ReLU
#     │       → class_logits [B,20,H,W]
#     │
#     └── regression tower
#             3×3 Conv → ReLU
#             3×3 Conv → ReLU
#             ├── box head → [B,4,H,W]
#             └── centerness head → [B,1,H,W]

class DetectionHead(nn.Module):
    def __init__(self, in_channels: int=128, num_classes: int=20  ,prior_probability: float=0.01) -> None:
        super().__init__()
        #建立分類3x3卷積層和回歸3x3卷積層的序列，包含兩個卷積層(參數初始化都是隨機)和ReLU激活函數
        self.class_tower= nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.box_tower= nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # 建立分類、回歸和中心度的卷積層，分別輸出類別數(此區塊對dog,cat等類別的分數、4個邊界框參數和1個中心度分數
        self.class_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.box_reg = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)  
        # 初始化Conv2d的weight和偏置，weight 從一個平均值 0、標準差 0.01 的常態分布隨機抽的小隨機值
        # 讓網路可以正常開始「學不同特徵」，相較於全0初始化，這樣可以避免神經元在訓練初期學到相同的特徵，導致模型表現不佳。
        # filter A → 對邊緣敏感
        # filter B → 對紋理敏感
        # filter C → 對形狀敏感
        if not 0.0 < prior_probability < 1.0:
            raise ValueError(
                "prior_probability must be between 0 and 1."
            )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01,
                )
                # 初始化bias為0 既然weight已經是小隨機值，我們通常沒有理由讓每個中間 Conv 一開始就額外偏向正或負。
                nn.init.zeros_(module.bias)
        # 當prior probability為0.01時，得到prior_bias ≈ -4.595
        prior_bias = math.log(
            prior_probability / (1.0 - prior_probability)
        )
        # 將class_logits的bias初始化為prior_bias
        # 這樣在訓練初期，模型對每個類別的預測概率接近於prior_probability=0.01，給模型一個假設:絕大多數 location 都不是物體。
        nn.init.constant_(
            self.class_logits.bias,
            prior_bias,
        )

    def forward(self, Pyramidfeatures: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        predictions = {}
        for level in ("p3", "p4", "p5"):
            features = Pyramidfeatures[level]
            #對p3、p4、p5的特徵圖分別進行分類和回歸操作
            class_features = self.class_tower(features)
            box_features = self.box_tower(features)
            #對分類特徵圖進行卷積操作，得到每塊區域的類別分數
            class_logits = self.class_logits(class_features)
            #對回歸特徵圖進行卷積操作，softplus(z)=log(1+e^z)，將raw box參數轉換為正值
            raw_box = self.box_reg(box_features)
            box_regression = F.softplus(raw_box)
            #對回歸特徵圖進行卷積操作，得到每塊區域的中心度分數
            centerness_logits = self.centerness(box_features)
            #回傳此層級的每塊區域分類分數、邊界框參數和中心度分數
            predictions[level] = {"class_logits": class_logits, "box_regression": box_regression, "centerness_logits": centerness_logits}
        return predictions
    