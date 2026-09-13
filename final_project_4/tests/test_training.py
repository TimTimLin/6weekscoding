import torch

from final_project_4.src.training import build_batch_targets
from final_project_4.src.detector import FCOSDetector
from final_project_4.src.training import train_one_step


REGRESSION_RANGES = {
    8: (0.0, 64.0),
    16: (64.0, 128.0),
    32: (128.0, float("inf")),
}


def test_build_batch_targets_shapes_and_alignment():
    # L = 3 個 prediction locations
    locations = torch.tensor([
        [4.0, 4.0],    # stride 8，應匹配小 box
        [8.0, 8.0],    # stride 16，應匹配大 box
        [20.0, 4.0],   # 背景
    ])

    location_strides = torch.tensor([8.0, 16.0, 8.0])

    dataset_targets = [
        {
            "boxes": torch.tensor([
                [0.0, 0.0, 16.0, 12.0],
            ]),
            "labels": torch.tensor([3], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([
                [0.0, 0.0, 100.0, 100.0],
            ]),
            "labels": torch.tensor([5], dtype=torch.int64),
        },
    ]

    assigned_targets = build_batch_targets(
        locations,
        location_strides,
        dataset_targets,
        REGRESSION_RANGES,
    )

    assert isinstance(assigned_targets, list)
    assert len(assigned_targets) == 2

    # 每張圖片的 output 都以 L 為主，不再是 Nᵢ
    assert assigned_targets[0]["labels"].shape == (3,)
    assert assigned_targets[0]["box_targets"].shape == (3, 4)
    assert assigned_targets[0]["centerness_targets"].shape == (3,)
    assert assigned_targets[0]["positive_mask"].shape == (3,)

    assert assigned_targets[1]["labels"].shape == (3,)
    assert assigned_targets[1]["box_targets"].shape == (3, 4)

    # image 0：location 0 匹配 label 3
    torch.testing.assert_close(
        assigned_targets[0]["labels"],
        torch.tensor([3, -1, -1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        assigned_targets[0]["positive_mask"],
        torch.tensor([True, False, False]),
    )
    torch.testing.assert_close(
        assigned_targets[0]["box_targets"][0],
        torch.tensor([4.0, 4.0, 12.0, 8.0]),
    )

    # image 1：location 1 匹配 label 5
    torch.testing.assert_close(
        assigned_targets[1]["labels"],
        torch.tensor([-1, 5, -1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        assigned_targets[1]["positive_mask"],
        torch.tensor([False, True, False]),
    )
    torch.testing.assert_close(
        assigned_targets[1]["box_targets"][1],
        torch.tensor([8.0, 8.0, 92.0, 92.0]),
    )


def test_build_batch_targets_empty_image():
    locations = torch.tensor([
        [4.0, 4.0],
        [8.0, 8.0],
    ])

    location_strides = torch.tensor([8.0, 16.0])

    dataset_targets = [
        {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
        }
    ]

    assigned_targets = build_batch_targets(
        locations,
        location_strides,
        dataset_targets,
        REGRESSION_RANGES,
    )

    target = assigned_targets[0]

    torch.testing.assert_close(
        target["labels"],
        torch.tensor([-1, -1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        target["box_targets"],
        torch.zeros((2, 4)),
    )
    torch.testing.assert_close(
        target["centerness_targets"],
        torch.zeros(2),
    )
    torch.testing.assert_close(
        target["positive_mask"],
        torch.tensor([False, False]),
    )
    torch.testing.assert_close(
        target["matched_gt_indices"],
        torch.tensor([-1, -1], dtype=torch.int64),
    )


def test_train_one_step_updates_parameters():
    torch.manual_seed(0)

    model = FCOSDetector(
        num_classes=20,
        fpn_channels=128,
        pretrained=False,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
    )

    images = torch.rand(2, 3, 320, 320)

    dataset_targets = [
        {
            "boxes": torch.tensor([
                [40.0, 40.0, 120.0, 120.0],
            ]),
            "labels": torch.tensor([3], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([
                [160.0, 100.0, 280.0, 260.0],
            ]),
            "labels": torch.tensor([7], dtype=torch.int64),
        },
    ]

    regression_ranges = {
        8: (0.0, 64.0),
        16: (64.0, 128.0),
        32: (128.0, float("inf")),
    }

    before = model.head.class_logits.weight.detach().clone()

    losses = train_one_step(
        model=model,
        optimizer=optimizer,
        images=images,
        dataset_targets=dataset_targets,
        regression_ranges=regression_ranges,
        num_classes=20,
    )

    expected_keys = {
        "classification_loss",
        "box_regression_loss",
        "centerness_loss",
        "loss",
    }

    assert set(losses.keys()) == expected_keys

    for value in losses.values():
        assert value.ndim == 0
        assert torch.isfinite(value)
        assert not value.requires_grad

    after = model.head.class_logits.weight.detach()

    assert not torch.equal(before, after)
    
from final_project_4.src.training import train_one_epoch


def test_train_one_epoch_one_batch():
    torch.manual_seed(0)

    model = FCOSDetector(
        num_classes=20,
        fpn_channels=128,
        pretrained=False,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
    )

    images = torch.rand(2, 3, 320, 320)

    dataset_targets = [
        {
            "boxes": torch.tensor([[40.0, 40.0, 120.0, 120.0]]),
            "labels": torch.tensor([3], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[160.0, 100.0, 280.0, 260.0]]),
            "labels": torch.tensor([7], dtype=torch.int64),
        },
    ]

    dataloader = [
        (images, dataset_targets),
    ]

    regression_ranges = {
        8: (0.0, 64.0),
        16: (64.0, 128.0),
        32: (128.0, float("inf")),
    }

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        regression_ranges=regression_ranges,
        num_classes=20,
        max_steps=1,
    )

    assert set(metrics.keys()) == {
        "classification_loss",
        "box_regression_loss",
        "centerness_loss",
        "loss",
    }

    for value in metrics.values():
        assert isinstance(value, float)
        assert torch.isfinite(torch.tensor(value))