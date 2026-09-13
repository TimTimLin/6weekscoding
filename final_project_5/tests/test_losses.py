
import torch
from final_project_5.src.losses import multiclass_dice_loss, cross_entropy_dice_loss
import math
def test_multiclass_dice_loss_uniform_prediction():
    logits = torch.zeros(
        (1, 3, 1, 3),
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor(
        [[
            [0, 1, 2]
        ]],
        dtype=torch.int64,
    )   
    dice_loss = multiclass_dice_loss(logits, targets)
    expected_loss = 1.0 - (2.0 * (1/3) + 1e-6) / (((1/3*3) + 1) + 1e-6)
    torch.testing.assert_close(
        dice_loss,
        torch.tensor(expected_loss, dtype=torch.float32),
        rtol=1e-5,
        atol=1e-6,
    )       
    assert dice_loss.ndim == 0, "Dice loss should be a scalar"  
    assert dice_loss.dtype == torch.float32, "Dice loss should have dtype torch.float32"
    assert torch.isfinite(dice_loss), "Dice loss should be finite"
    dice_loss.backward()

    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))  

def test_multiclass_dice_loss_rewards_correct_overlap():
    correct_logits = torch.tensor(
        [
            [
                [[10.0, -10.0, -10.0]],  # class 0
                [[-10.0, 10.0, -10.0]],  # class 1
                [[-10.0, -10.0, 10.0]],  # class 2
            ]
        ]
    )
    targets = torch.tensor([[[0, 1, 2]]], dtype=torch.int64)
    # 接近完美的预测，loss 应该接近 0
    correct_dice_loss = multiclass_dice_loss(correct_logits, targets)
    wrong_logits = torch.tensor(
        [
            [
                [[-10.0, -10.0, 10.0]],  # class 0
                [[10.0, -10.0, -10.0]],  # class 1
                [[-10.0, 10.0, -10.0]],  # class 2
            ]
        ]
    )
    # 完全错误的预测，loss 应该接近 1
    wrong_dice_loss = multiclass_dice_loss(wrong_logits, targets)
    assert correct_dice_loss < 1e-4
    assert wrong_dice_loss > 0.99
    assert correct_dice_loss < wrong_dice_loss
    
def test_cross_entropy_dice_loss_combines_components():
    logits = torch.zeros(
        (1, 3, 1, 3),
        dtype=torch.float32,
        requires_grad=True,
    )

    targets = torch.tensor(
        [[[0, 1, 2]]],
        dtype=torch.int64,
    )

    dice_weight = 0.5
    smooth = 1e-6
    cross_entropy_dice = cross_entropy_dice_loss(
        logits,
        targets,
        dice_weight=dice_weight,
        smooth=smooth,
    )
    expected_ce = math.log(3.0)

    expected_dice = 1.0 - (
        (2.0 / 3.0 + smooth)
        / (2.0 + smooth)
    )

    expected_total = torch.tensor(
        expected_ce
        + dice_weight * expected_dice
    , dtype=torch.float32)
    torch.testing.assert_close(
        cross_entropy_dice,
        expected_total,
        rtol=1e-5,
        atol=1e-6,
    )
    assert cross_entropy_dice.ndim == 0, "Total loss should be a scalar"
    assert torch.isfinite(cross_entropy_dice), "Total loss should be finite"    
    cross_entropy_dice.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))