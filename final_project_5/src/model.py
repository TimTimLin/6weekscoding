
import torch

from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)
# x
# │
# ├─ enc1: DoubleConv(3,32)   → skip1
# ├─ MaxPool2d(2)
# ├─ enc2: DoubleConv(32,64)  → skip2
# ├─ MaxPool2d(2)
# ├─ enc3: DoubleConv(64,128) → skip3
# ├─ MaxPool2d(2)
# └─ bottleneck: DoubleConv(128,256)
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels) #double conv (3, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2) #double conv (32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4) #double conv (64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8) # double conv (128, 256) 

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool1(skip1))
        skip3 = self.enc3(self.pool2(skip2))
        bottleneck_out = self.bottleneck(self.pool3(skip3))
        return bottleneck_out, skip3, skip2, skip1
    
# bottleneck_out
# │
# ├─ upconv3: ConvTranspose2d(256,128) → spatial size ×2
# ├─ concat skip3: 128 + 128 = 256 channels
# ├─ dec3: DoubleConv(256,128)
# │
# ├─ upconv2: ConvTranspose2d(128,64) → spatial size ×2
# ├─ concat skip2: 64 + 64 = 128 channels
# ├─ dec2: DoubleConv(128,64)
# │
# ├─ upconv1: ConvTranspose2d(64,32) → spatial size ×2
# ├─ concat skip1: 32 + 32 = 64 channels
# └─ dec1: DoubleConv(64,32) → decoder_out
class UNetDecoder(nn.Module):
    def __init__(self, base_channels=32) -> None:
        super().__init__()
        # ConvTranspose2d 先學習性地把 decoder feature 放大到和 skip 相同解析度，讓兩者能 concat；真正提供原本精細位置資訊的主要是 skip connection。
        # 也類似upsampling + conv的概念方式。
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        
    def forward(self, bottleneck_out: torch.Tensor, skip3: torch.Tensor, skip2: torch.Tensor, skip1: torch.Tensor) -> torch.Tensor:
        up3 = self.upconv3(bottleneck_out)
        x = torch.cat((up3, skip3), dim=1)  
        x = self.dec3(x)
        
        up2 = self.upconv2(x)
        x = torch.cat((up2, skip2), dim=1)
        x = self.dec2(x)

        up1 = self.upconv1(x)
        x = torch.cat((up1, skip1), dim=1)
        decoder_out = self.dec1(x)

        return decoder_out

# 把 encoder 和 decoder 組合起來，最後再接一個 1x1 conv，把 base_channels 的 feature map 壓縮成 num_classes 個 channel，作為每個 pixel 的在各個類別上的分數
class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_channels=32) -> None:
        super().__init__()
        self.encoder = UNetEncoder(in_channels, base_channels)
        self.decoder = UNetDecoder(base_channels)
        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_out, skip3, skip2, skip1 = self.encoder(x)
        decoder_out = self.decoder(bottleneck_out, skip3, skip2, skip1)
        output = self.classifier(decoder_out)
        return output