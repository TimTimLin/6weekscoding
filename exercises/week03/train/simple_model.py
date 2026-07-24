import torch
from torch import nn
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__() #繼承nn.Module的初始化方法 eg: model.train()、model.eval()、model.parameters()等方法
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 32 * 3, 10)  # Assuming input images are of shape (3, 32, 32) and we have 10 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x) #將圖像展平為一維向量 (16, 32 * 32 * 3)
        x = self.linear(x) # 將展平後的向量輸入線性層，得到 (16, 10) 的輸出，表示每個圖像對應 10 個類別的 logits
        return x


