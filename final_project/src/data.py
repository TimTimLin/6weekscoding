from pathlib import Path

import torch

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(batch_size: int = 64, seed: int = 42):
    # Define the root directory for the data 
    # __file__ 代表當前文件的路徑，resolve() 會返回該路徑的絕對路徑，parents[1] 會返回該路徑的父目錄的父目錄，最後再加上 "data" 就是我們要存放數據集的目錄
    data_root = Path(__file__).resolve().parents[1] / "data"
    # 定義transform，將圖像轉成pytorch的格式，
    transform = transforms.ToTensor()
    #建立FashionMNIST數據集的訓練集，預設60000個樣本
    # test_dataset 不要 random_split
    dataset = datasets.FashionMNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )
        #建立FashionMNIST數據集的測試集，預設10000個樣本
    test_dataset = datasets.FashionMNIST(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )
    train_subset, val_subset = random_split(dataset, [54000, 6000], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False)    
    validation_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, validation_loader, test_loader

    