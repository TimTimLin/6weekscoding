import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ToyImageDataset(Dataset):
    def __init__(
        self,
        images: torch.Tensor, #預期 images: torch.Tensor, #預期輸入為一個形狀為 (N, C, H, W) 的張量，表示 N 張圖像，每張圖像有 C 個通道，高度為 H，寬度為 W。
        labels: torch.Tensor, #預期 labels: torch.Tensor, #預期輸入為一個形狀為 (N,) 的張量，表示 N 張圖像的標籤。
    ):
        self.images = images 
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


images = torch.rand(100, 3, 32, 32)
labels = torch.randint(0, 10, (100,))

dataset = ToyImageDataset(images, labels)



print(len(dataset))
print(dataset[0][0].shape)  #dataset[0][0] 是第一張圖像，dataset[0][0].shape 是其形狀，應該是 (3, 32, 32)
print(dataset[0][1])        #dataset[0][1] 是第一張圖像的標籤，應該是一個整數，表示該圖像的類別。
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False) 
batch_images, batch_labels = next(iter(dataloader))
print(len(dataloader))  #len(dataloader) 是 dataloader 中的批次
print(batch_images.shape)  #batch_images.shape 是批次中圖像的形狀，應該是 (16, 3, 32, 32)
print(batch_labels.shape)  #batch_labels.shape 是批次中標籤的形狀，應該是 (16,)