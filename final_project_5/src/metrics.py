

import torch


#                     Prediction
#                  0       1       2
# Target 0       C[0,0] C[0,1] C[0,2]
# Target 1       C[1,0] C[1,1] C[1,2]
# Target 2       C[2,0] C[2,1] C[2,2]
def confusion_matrix_from_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    # 把 predictions 和 targets 轉成一維向量
    flatten_predictions = predictions.reshape(-1)
    flatten_targets = targets.reshape(-1)
    # 把 predictions 和 targets 轉成一個編碼的索引，這樣可以用 bincount 計算每個類別的數量
    # target → pred       encoded
    # 0 → 0               0
    # 0 → 1               1
    # 0 → 1               1
    # 1 → 1               4
    # 1 → 2               5
    # 2 → 2               8
    encoded_indices = flatten_targets * num_classes + flatten_predictions
    # 使用 bincount 計算encoded_indices中每個數字的數量，並將其重塑為混淆矩陣
    # encoded_indices =[0, 1, 1, 4, 5, 8]
    # 數字 0 → 1 次
    # 數字 1 → 2 次
    # 數字 2 → 0 次
    # 數字 3 → 0 次
    # 數字 4 → 1 次
    # 數字 5 → 1 次
    # 數字 6 → 0 次
    # 數字 7 → 0 次
    # 數字 8 → 1 次
    # counts = [1, 2, 0, 0, 1, 1, 0, 0, 1]
    counts = torch.bincount(encoded_indices, minlength=num_classes * num_classes)
    # 將 counts 重塑為混淆矩陣
    confusion_matrix = counts.reshape(num_classes, num_classes)
    return confusion_matrix

def iou_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        confusion_matrix.ndim != 2
        or confusion_matrix.shape[0] != confusion_matrix.shape[1]
    ):
        raise ValueError("Confusion matrix must be a square matrix.")
    confusion_matrix = confusion_matrix.to(torch.float64)
    #                         Prediction
    #                 0       1       2
    # Target 0         1       2       0
    # Target 1         0       1       1
    # Target 2         0       0       1
    # TPk = matrix[k,k]
    # GTk = row k 的總和
    # Predk = column k 的總和

    # FNk = GTk - TPk
    # FPk = Predk - TPk
    true_positives = torch.diagonal(confusion_matrix)
    target_counts = confusion_matrix.sum(dim=1)
    prediction_counts = confusion_matrix.sum(dim=0)
    union = target_counts + prediction_counts - true_positives      
    # 只計算有出現的類別的 IoU，避免除以 0
    valid_classes = union > 0
    # 先把 per_class_iou 初始化為 NaN，這樣可以避免除以 0 的情況
    per_class_iou = torch.full_like(union,fill_value=float("nan"))
    # ioU = TPk / (TPk + FPk + FNk) = TPk / union
    # 計算 per_class_iou 只對 valid_classes 計算，避免除以 0
    per_class_iou[valid_classes] = true_positives[valid_classes] / union[valid_classes]
    mean_iou = per_class_iou[valid_classes].mean()
    return per_class_iou, mean_iou