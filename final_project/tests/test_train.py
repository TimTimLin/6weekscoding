import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from final_project.src.model import FashionCNN
from final_project.src.train import train_one_epoch


def test_train_one_epoch_updates_model():
    torch.manual_seed(0)

    images = torch.rand(8, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=False,
    )

    model = FashionCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    # 紀錄模型在訓練前的權重
    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    loss, accuracy = train_one_epoch(
        model,
        loader,
        loss_fn,
        optimizer,
    )
    # 紀錄模型在訓練後的權重
    after = list(model.parameters())
    
    # 確認損失值是有限的，準確率在0到100之間，並且模型的權重已經更新
    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= accuracy <= 100.0
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, after)
    )