from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)




def get_dataloaders(batch_size: int = 64, seed: int = 42,train_subset_size: int = None, validation_subset_size: int = None):
    """Return train, validation, and official CIFAR-10 test loaders."""
    # The original data directory contains a partially extracted archive with
    # restrictive Windows permissions. Use the clean local extraction instead.
    data_root = Path(__file__).resolve().parents[1] / "data"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    full_train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )
    train_dataset, validation_dataset = random_split(
            full_train_dataset,
            [45_000, 5_000],
            generator=torch.Generator().manual_seed(seed),
        )
    #當有指定train_subset_size或validation_subset_size時，從train_dataset和validation_dataset中隨機選取指定數量的樣本
    if train_subset_size is not None:
        generator = torch.Generator().manual_seed(seed + 1)
        indices = torch.randperm(
            len(train_dataset),
            generator=generator,
        )[:train_subset_size]
        train_dataset = Subset(train_dataset, indices.tolist())

    if validation_subset_size is not None:
        generator = torch.Generator().manual_seed(seed + 2)
        indices = torch.randperm(
            len(validation_dataset),
            generator=generator,
        )[:validation_subset_size]
        validation_dataset = Subset(validation_dataset, indices.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, validation_loader, test_loader
