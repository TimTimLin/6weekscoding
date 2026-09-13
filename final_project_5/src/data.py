
import torch
import numpy as np
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Subset
def remap_trimap(raw_mask) -> torch.Tensor:
    # 將 PIL Image 轉換為 numpy array
    mask_array = np.array(raw_mask,copy=True)
    if not np.all(np.isin(mask_array, [1, 2, 3])):
        raise ValueError("Input mask contains values other than 1, 2, or 3.")
    # 建立一個新的空白 mask
    remapped_mask = np.zeros_like(mask_array, dtype=np.int64)

    # 將foreground (1) 和background (2) notclassified (3)分別映射到新的類別 
    # The mapping is as follows:
    # Original Value | New Value
    # 1 | 0 (foreground)
    # 2 | 1 (background)
    # 3 | 2 (not classified)
    remapped_mask[mask_array == 1] = 0  
    remapped_mask[mask_array == 2] = 1  
    remapped_mask[mask_array == 3] = 2

    # 將 numpy array 轉換為 torch tensor
    return torch.from_numpy(remapped_mask)

# 建立OxfordIIITPet資料集的函數，並將joint_transform和download參數傳遞給OxfordIIITPet類別
def build_oxford_pet_dataset(root,split,joint_transform,download=False):  
    return OxfordIIITPet(root=root,split=split,target_types="segmentation",transforms=joint_transform,download=download)

def split_trainval_indices(
    num_samples: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    val_size = int(num_samples * val_fraction)
    shuffled_indices = torch.randperm(num_samples, generator=generator).tolist()
    train_indices = shuffled_indices[val_size:]
    val_indices = shuffled_indices[:val_size]
    return train_indices, val_indices

def build_trainval_datasets(
    root,
    val_fraction: float,
    seed: int,
    train_transform,
    val_transform,
    download=False,
) -> tuple[Subset, Subset]:
    train_base_dataset = build_oxford_pet_dataset(
        root=root, split="trainval", joint_transform=train_transform, download=download
    )
    val_base_dataset = build_oxford_pet_dataset(
        root=root, split="trainval", joint_transform=val_transform, download=download
    )
    if len(train_base_dataset) != len(val_base_dataset):
        raise ValueError(
            "Train and validation base datasets must have the same length."
        )
    train_indices, val_indices = split_trainval_indices(
        num_samples=len(train_base_dataset), val_fraction=val_fraction, seed=seed
    )
    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)
    return train_dataset, val_dataset