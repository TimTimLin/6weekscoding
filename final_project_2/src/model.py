import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.features=nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
                
    def forward(self, x):  
            #只單獨使用features的輸出，沒有加上shortcut
            out = self.features(x) 
            return out

class PreActResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.features=nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        if in_channels == out_channels and stride == 1:
            #在不需要改变通道数和尺寸的情况下，shortcut直接使用恒等映射
            self.shortcut = nn.Identity() 
        else:
            #在需要改变通道数或尺寸的情况下，shortcut使用1x1卷积进行映射
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)   
            
    def forward(self, x):  
        identity = self.shortcut(x) #決定identity的映射方式
        out = self.features(x) 
        out = out + identity
        return out
class CIFARResNet(nn.Module):
    def __init__(self, num_classes=10,in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = PreActResidualBlock(32, 32, stride=1)
        self.layer2 = PreActResidualBlock(32, 32, stride=1)
        self.layer3 = PreActResidualBlock(32, 64, stride=2)
        self.layer4 = PreActResidualBlock(64, 64, stride=1)
        self.layer5 = PreActResidualBlock(64, 128, stride=2)
        self.layer6 = PreActResidualBlock(128, 128, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 每個 channel 的 64 個值取平均 8*8->1*1
        self.fc = nn.Linear(128*1*1, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
class CIFARPlainDeepNet(nn.Module):
        def __init__(self, num_classes=10,in_channels=3):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
            self.layer1 = PlainBlock(32, 32, stride=1)
            self.layer2 = PlainBlock(32, 32, stride=1)
            self.layer3 = PlainBlock(32, 64, stride=2)
            self.layer4 = PlainBlock(64, 64, stride=1)
            self.layer5 = PlainBlock(64, 128, stride=2)
            self.layer6 = PlainBlock(128, 128, stride=1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 每個 channel 的 64 個值取平均 8*8->1*1
            self.fc = nn.Linear(128*1*1, num_classes)
    
        def forward(self, x):
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x    