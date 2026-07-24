import torch
from torch import nn

class TinyCNN(nn.Module):
    def __init__(self,num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier=nn.Linear(
            in_features=16*8*8, 
            out_features=num_classes
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # Flatten the output of the feature extractor
        x = self.classifier(x)
        return x