import torch
from final_project.src.data import get_dataloaders

def test_fashion_mnist_dataloaders():
    train_loader, validation_loader, test_loader = get_dataloaders(batch_size=8,seed=42)

    images, labels = next(iter(train_loader))

    #確認數據集的大小是否正確
    assert len(train_loader.dataset) == 54000
    assert len(validation_loader.dataset) == 6000
    assert len(test_loader.dataset) == 10000

    # 確認image和label的shape和type是否正確
    assert images.shape == (8, 1, 28, 28)
    assert labels.shape == (8,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    # 確認圖像的像素值是否在0到1之間
    assert 0.0 <= images.min().item() <= 1.0
    assert 0.0 <= images.max().item() <= 1.0

def test_split_is_reproducible():
    train_loader_a, val_loader_a, _ = get_dataloaders(
        batch_size=4,
        seed=42,
    )

    train_loader_b, val_loader_b, _ = get_dataloaders(
        batch_size=4,
        seed=42,
    )
    # 確認兩次調用 get_dataloaders 時，訓練集和驗證集的索引是否相同
    assert train_loader_a.dataset.indices == train_loader_b.dataset.indices
    assert val_loader_a.dataset.indices == val_loader_b.dataset.indices 
