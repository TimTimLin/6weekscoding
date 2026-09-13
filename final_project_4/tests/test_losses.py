import torch

from final_project_4.src.losses import (box_regression_loss, centerness_loss, 
sigmoid_focal_loss, labels_to_one_hot, compute_detection_loss)


def test_sigmoid_focal_loss_scalar_and_finite():
    logits = torch.zeros(2, 4, 3)
    targets = torch.zeros(2, 4, 3)
    # 指定一些目標值為1，模擬正樣本
    targets[0, 1, 2] = 1.0
    targets[1, 3, 0] = 1.0

    loss = sigmoid_focal_loss(
        logits,
        targets,
        normalizer=2.0,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_sigmoid_focal_loss_one_element_value():
    logits = torch.tensor([[[0.0]]])
    targets = torch.tensor([[[1.0]]])

    loss = sigmoid_focal_loss(
        logits,
        targets,
        alpha=0.25,
        gamma=2.0,
        normalizer=1.0,
    )
    # 計算預期的 focal loss 值，使用公式計算
    # p_t = sigmoid(logits) = 0.5
    expected = 0.25 * (1.0 - 0.5) ** 2 * torch.log(
        torch.tensor(2.0)
    )

    torch.testing.assert_close(loss, expected)
    
def test_sigmoid_focal_loss_negative_alpha_weight():
    logits = torch.tensor([[[0.0]]])
    targets = torch.tensor([[[0.0]]])

    loss = sigmoid_focal_loss(
        logits,
        targets,
        alpha=0.25,
        gamma=2.0,
        normalizer=1.0,
    )

    expected = (
        0.75
        * (1.0 - 0.5) ** 2
        * torch.log(torch.tensor(2.0))
    )

    torch.testing.assert_close(loss, expected)
    
def test_labels_to_one_hot():
    labels = torch.tensor([
        [11, -1, 0],
    ])

    targets = labels_to_one_hot(
        labels,
        num_classes=20,
    )

    assert targets.shape == (1, 3, 20)
    assert targets.dtype == torch.float32

    assert targets[0, 0, 11] == 1.0
    assert targets[0, 2, 0] == 1.0

    torch.testing.assert_close(
        targets[0, 1],
        torch.zeros(20),
    )

def test_box_regression_loss_ignores_background():
    predicted = torch.tensor([
        [
            [5.0, 5.0, 10.0, 10.0],
            [999.0, 999.0, 999.0, 999.0],
        ]
    ])

    target = torch.tensor([
        [
            [4.0, 4.0, 8.0, 8.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ])
    locations = torch.tensor([
        [4.0, 4.0],
        [8.0, 8.0],
    ])                                                                  
    positive_mask = torch.tensor([
        [True, False]
    ])

    loss_1 = box_regression_loss(
        predicted_ltrb=predicted,
        target_ltrb=target,
        locations=locations,
        positive_mask=positive_mask,
        normalizer=1.0,
    )

    predicted_changed = predicted.clone()
    predicted_changed[0, 1] = -999.0

    loss_2 = box_regression_loss(
        predicted_ltrb=predicted_changed,
        target_ltrb=target,
        locations=locations,
        positive_mask=positive_mask,
        normalizer=1.0,
    )

    torch.testing.assert_close(loss_1, loss_2)
    assert loss_1.ndim == 0
    assert torch.isfinite(loss_1)


def test_box_regression_loss_giou_value():
    locations = torch.tensor([
        [5.0, 5.0],
    ])

    # Prediction 解碼後為 [4, 4, 6, 6]，面積為 4。
    predicted = torch.tensor([[
        [1.0, 1.0, 1.0, 1.0],
    ]])

    # Target 解碼後為 [0, 0, 10, 10]，面積為 100。
    target = torch.tensor([[
        [5.0, 5.0, 5.0, 5.0],
    ]])

    positive_mask = torch.tensor([
        [True],
    ])

    loss = box_regression_loss(
        predicted_ltrb=predicted,
        target_ltrb=target,
        locations=locations,
        positive_mask=positive_mask,
        normalizer=1.0,
    )

    # Prediction 完全位於 target 內，因此：
    # IoU = GIoU = 4 / 100 = 0.04，loss = 1 - 0.04 = 0.96。
    torch.testing.assert_close(
        loss,
        torch.tensor(0.96),
    )

def test_centerness_loss_ignores_background():
    logits = torch.tensor([
        [0.0, 100.0, -100.0]
    ])

    targets = torch.tensor([
        [1.0, 0.0, 0.5]
    ])

    positive_mask = torch.tensor([
        [True, False, True]
    ])

    loss = centerness_loss(
        logits,
        targets,
        positive_mask,
        normalizer=2.0,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    
def test_losses_with_no_positive_are_zero():
    predicted = torch.randn(1, 4, 4)
    target = torch.zeros(1, 4, 4)
    locations = torch.tensor([
        [4.0, 4.0],
        [12.0, 4.0],
        [4.0, 12.0],
        [12.0, 12.0],
    ])
    logits = torch.randn(1, 4)
    center_target = torch.zeros(1, 4)
    positive_mask = torch.zeros(1, 4, dtype=torch.bool)

    box_loss = box_regression_loss(
        predicted_ltrb=predicted,
        target_ltrb=target,
        locations=locations,
        positive_mask=positive_mask,
        normalizer=1.0,
    )

    center_loss = centerness_loss(
        logits,
        center_target,
        positive_mask,
        normalizer=1.0,
    )

    assert box_loss.item() == 0.0
    assert center_loss.item() == 0.0
    

def test_compute_detection_loss():
    targets = [
        {
            "labels": torch.tensor([-1, 2, -1], dtype=torch.int64),
            "box_targets": torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [4.0, 4.0, 4.0, 4.0],
                [0.0, 0.0, 0.0, 0.0],
            ]),
            "centerness_targets": torch.tensor([
                0.0,
                0.5,
                0.0,
            ]),
            "positive_mask": torch.tensor([
                False,
                True,
                False,
            ]),
        },
        {
            "labels": torch.tensor([-1, 5, -1], dtype=torch.int64),
            "box_targets": torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [4.0, 4.0, 4.0, 4.0],
                [0.0, 0.0, 0.0, 0.0],
            ]),
            "centerness_targets": torch.tensor([
                0.0,
                1.0,
                0.0,
            ]),
            "positive_mask": torch.tensor([
                False,
                True,
                False,
            ]),
        },
    ]

    predictions = {
        "class_logits": torch.zeros(
            2, 3, 20, requires_grad=True
        ),
        "box_regression": torch.ones(
            2, 3, 4, requires_grad=True
        ),
        "centerness_logits": torch.zeros(
            2, 3, requires_grad=True
        ),
        "locations": torch.tensor([
            [4.0, 4.0],
            [12.0, 12.0],
            [20.0, 20.0],
        ]),
    }

    losses = compute_detection_loss(
        predictions,
        targets,
        num_classes=20,
        alpha=0.25,
        gamma=2.0,
    )

    expected_keys = {
        "loss",
        "classification_loss",
        "box_regression_loss",
        "centerness_loss",
    }

    assert set(losses.keys()) == expected_keys

    for loss in losses.values():
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss).item()

    expected_total = (
        losses["classification_loss"]
        + losses["box_regression_loss"]
        + losses["centerness_loss"]
    )

    torch.testing.assert_close(
        losses["loss"],
        expected_total,
    )

    losses["loss"].backward()

    assert predictions["class_logits"].grad is not None
    assert predictions["box_regression"].grad is not None
    assert predictions["centerness_logits"].grad is not None
