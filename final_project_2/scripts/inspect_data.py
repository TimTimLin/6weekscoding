from pathlib import Path
import torch
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms

def main() -> None :
    data_root = Path(__file__).resolve().parents[1] / "data"
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    images, labels = next(iter(loader))
    print(f"Images shape: {images.shape}, Images dtype: {images.dtype}, Images min: {images.min()}, Images max: {images.max()}")
    print(f"Labels shape: {labels.shape}, Labels dtype: {labels.dtype}, Labels unique values: {labels.unique()}")
    print(f'length of dataset: {len(dataset)}')
if __name__ == "__main__":
    main()
    