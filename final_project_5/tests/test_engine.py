
import math
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from final_project_5.src.engine import train_one_epoch, train_step , evaluate

def test_train_step_updates_parameters():
    torch.manual_seed(0)
    images = torch.randn(2, 3, 4, 5)

    masks = torch.randint(
        low=0,
        high=3,
        size=(2, 4, 5),
        dtype=torch.int64,
    )
    model = nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=1,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]
    loss = train_step(
        model,
        images,
        masks,
        optimizer,
        loss_fn,
    )       
    assert isinstance(loss, float), "Loss should be a Python float"
    assert math.isfinite(loss), "Loss should be finite"
    assert model.training, "Model should be in training mode after train_step"
    parameters_changed = any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            model.parameters(),
        )
    )

    assert parameters_changed
    
def test_train_one_epoch_uses_sample_weighted_loss():
    images = torch.zeros(5, 1, 1, 1)

    masks = torch.tensor(
        [1, 1, 2, 2, 4],
        dtype=torch.int64,
    ).reshape(5, 1, 1)
    dataset = torch.utils.data.TensorDataset(images, masks)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )
    
    model = nn.Conv2d(1, 1, kernel_size=1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0,
    )
    
    def loss_fn(logits, masks):
        return (
            logits.sum() * 0.0
            + masks.float().mean()
        )
    epoch_loss = train_one_epoch(
        model,
        dataloader,
        optimizer,
        loss_fn,
        device=torch.device("cpu")
    )
    epoch_loss_expected = masks.float().mean().item()
    assert isinstance(epoch_loss, float), "Epoch loss should be a Python float"
    assert math.isfinite(epoch_loss), "Epoch loss should be finite"
    assert math.isclose(epoch_loss, epoch_loss_expected, rel_tol=1e-5), "Epoch loss should match the expected value"
    assert model.training, "Model should be in training mode after train_one_epoch"
    
def test_train_one_epoch_rejects_empty_dataloader():
    images = torch.empty(0, 1, 1, 1)

    masks = torch.empty(
        0, 1, 1,
        dtype=torch.int64,
    )

    dataset = torch.utils.data.TensorDataset(
        images,
        masks,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
    )
    model = nn.Conv2d(1, 1, kernel_size=1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )
    with pytest.raises(ValueError):
        train_one_epoch(
            model,
            dataloader,
            optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )
        
def test_evaluate_accumulates_all_batches_without_gradients():
    class ScaleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.grad_enabled_during_forward = None
        def forward(self, images):
            self.grad_enabled_during_forward = torch.is_grad_enabled()
            return images * self.scale
        
    model = ScaleModel()
    wanted_predictions = torch.tensor(
        [
            [[0, 1, 1]],
            [[1, 2, 2]],
        ],
        dtype=torch.int64,
    )
    images = F.one_hot(
        wanted_predictions,
        num_classes=3,
    ).permute(0, 3, 1, 2).float() * 10.0
    targets = torch.tensor(
        [
            [[0, 0, 0]],
            [[1, 1, 2]],
        ],
        dtype=torch.int64,
    )
    expected_loss = F.cross_entropy(
        images,
        targets,
    ).item()    
    expected_confusion = torch.tensor(
        [
            [1, 2, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=torch.int64,
    )
    parameter_before = model.scale.detach().clone()
    result = evaluate(
        model=model,
        dataloader=torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(images, targets),
            batch_size=1,
        ),
        loss_fn=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        num_classes=3,
    )
    assert isinstance(result, dict), "Result should be a dictionary"
    assert math.isclose(result["loss"], expected_loss, rel_tol=1e-5), "Result loss should match the expected value"
    assert torch.equal(result["confusion_matrix"], expected_confusion), "Result confusion should match the expected value"
    torch.testing.assert_close(
        result["per_class_iou"],
        torch.tensor(
            [1.0 / 3.0, 1.0 / 4.0, 1.0 / 2.0],
            dtype=torch.float64,
        ),  
        rtol=1e-5,
        atol=1e-8,
    )
    assert math.isclose(result["mean_iou"], 13.0 / 36.0, rel_tol=1e-5), "Result mean IoU should match the expected value"
    assert model.training is False, "Model should be in evaluation mode after evaluation"
    assert model.scale.detach().equal(parameter_before), "Model parameters should not change during evaluation"
    assert model.scale.grad is None, "Model parameters should not have gradients during evaluation"
    assert model.grad_enabled_during_forward is False, "Model should not have gradients enabled during evaluation"          
