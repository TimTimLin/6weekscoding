


import torch
from torch.nn import functional as F

def multiclass_dice_loss(logits: torch.Tensor,targets: torch.Tensor,smooth: float = 1e-6) -> torch.Tensor:
    # 將 logits 轉換為 softmax 機率分佈 
    #              class 0   class 1   class 2
    # pixel 0         0         0         0    #每個class的分數
    # pixel 1         0         0         0
    # pixel 2         0         0         0
    #              class 0   class 1   class 2
    # pixel 0        1/3       1/3       1/3   #每個class的機率
    # pixel 1        1/3       1/3       1/3
    # pixel 2        1/3       1/3       1/3
    prediction_probs = torch.softmax(logits, dim=1)
    # GT 轉成 one-hot encoding
    #                class   
    # pixel 0        0      
    # pixel 1        1      
    # pixel 2        2     
    #              class 0   class 1   class 2
    # pixel 0        1       0       0
    # pixel 1        0       1       0
    # pixel 2        0       0       1
    one_hot_targets = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).to(dtype=prediction_probs.dtype)
#     計算 intersection 
#     prediction class 0:   [1/3, 1/3, 1/3]
#     GT class 0:           [ 1 ,  0 ,  0 ]
#                             ↓    ↓    ↓
#     multiply:             [1/3,  0 ,  0 ]
    intersection = torch.sum(prediction_probs * one_hot_targets, dim=(0, 2, 3))
    denominator = torch.sum(prediction_probs, dim=(0, 2, 3)) + torch.sum(one_hot_targets, dim=(0, 2, 3))
    # 計算dice score  = (2 * intersection + smooth) / (denominator + smooth)
    dice_score = (2.0 * intersection + smooth) / (denominator + smooth)
    # 計算 dice loss = 1 - dice score
    dice_loss = 1.0 - dice_score
    return dice_loss.mean()

# 把 cross_entropy_loss 和 dice_loss 結合起來為一個新的 loss function
# Total Loss = Cross_Entropy_loss + dice_weight × Dice_Loss
def cross_entropy_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_weight: float = 1.0,
    smooth: float = 1e-6,
) -> torch.Tensor:
    ce_loss = F.cross_entropy(logits, targets)
    dice_loss = multiclass_dice_loss(logits, targets, smooth=smooth)
    total_loss = ce_loss + dice_weight * dice_loss
    return total_loss