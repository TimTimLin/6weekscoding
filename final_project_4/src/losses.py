import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss
from .boxes import decode_boxes
#相較binary cross entropy  focal loss能夠更好地處理類別不平衡的問題，對於難以分類的樣本給予更高的權重
#大量容易分類的 background→ loss 佔比過大→ 模型只學會預測全部 background不知道如何去辨識少數類別的樣本
#              Focal Loss
#                   │
#         ┌─────────┴─────────┐
#         ↓                   ↓
#    alpha_t              focal factor
#    平衡正負              平衡難易
#         │                   │
#    positive/negative      easy/hard
#    權重                    權重
def sigmoid_focal_loss(logits: torch.Tensor,targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    normalizer: float = 1.0,
) -> torch.Tensor:
    #正常化的二元交叉熵損失函數，使用logits和targets計算損失
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    #計算每個樣本的概率，使用sigmoid函數將logits轉換為概率值
    probabilities = torch.sigmoid(logits)
    #if targets=1, p_t=probabilities; if targets=0, p_t=1-probabilities
    #假設模型很確定這個區塊是背景，對其他20個class的預測值都很低，這時候p_t會接近1，(1-p_t)會接近0，focal loss的權重就會很小，對於容易分類的樣本給予較小的權重
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    #計算focal loss的權重，對於難以分類的樣本給予更高的權重
    focal_weight = (1 - p_t) ** gamma
    #if targets=1, alpha_factor=alpha; if targets=0, alpha_factor=1-alpha
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    #計算最終的focal loss，將alpha_factor、focal_weight和bce_loss相乘，並對損失進行加總和正規化
    loss = alpha_factor * focal_weight * bce_loss
    return loss.sum() / normalizer

def labels_to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    #找出不是背景的樣本，這些樣本的標籤值大於等於0 
    # example: labels = [0, 1, 2, -1] → positive_mask = [True, True, True, False]
    positive_mask = labels >= 0
    # 把背景的標籤值設為0，避免在one-hot編碼時出現負數索引
    safe_labels = labels.clamp(min=0)
    # 將標籤轉換為one-hot編碼，使用scatter_方法將對應位置設為1
    # example: safe_labels = [0, 1, 2, 0] → one_hot = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
    one_hot = F.one_hot(safe_labels, num_classes=num_classes).to(torch.float32)
    #把background的那列設為0，避免對loss計算造成影響 [true, False, False] → [[true
    #                                                                       flase, 
    #                                                                       false]] 然後和onehot相乘
    # Background 的 one-hot row 全為 0。
    # 這不代表它不參與 loss；它會對所有 C 個類別產生 negative
    # focal-loss targets，讓模型學會「這個 location 沒有任何物體」。
    one_hot = one_hot * positive_mask.unsqueeze(-1).to(torch.float32)
    return one_hot

def box_regression_loss(
    predicted_ltrb: torch.Tensor,  # [B,L,4]
    target_ltrb: torch.Tensor,     # [B,L,4]
    locations: torch.Tensor,       # [L,2]
    positive_mask: torch.Tensor,   # [B,L]
    normalizer: float = 1.0,
) -> torch.Tensor:                 # scalar []
    #使LTRB的預測值和目標值轉換為實際的邊界框座標，方便計算損失
    decoded_pred_boxes = decode_boxes(locations, predicted_ltrb)
    decoded_target_boxes = decode_boxes(locations, target_ltrb)
    # 留下正樣本的預測邊界框
    positive_pred= predicted_ltrb[positive_mask]
    # 如果沒有正樣本，返回0的損失，避免除以0的情況
    if positive_pred.numel() == 0:
        return predicted_ltrb.sum() * 0.0
    # 計算正樣本的gIoU損失，衡量預測邊界框和目標邊界框之間的差距
    generalized_iou_loss = generalized_box_iou_loss(
        decoded_pred_boxes[positive_mask],
        decoded_target_boxes[positive_mask],reduction="sum"
    )

    return generalized_iou_loss/ max(normalizer, 1.0)
def centerness_loss(predicted_centerness: torch.Tensor, target_centerness: torch.Tensor,
    positive_mask: torch.Tensor, normalizer: float = 1.0) -> torch.Tensor:
    #留下正樣本的預測centerness和目標centerness，計算binary cross entropy loss，衡量中心度的預測值和目標值之間的差距 
    positive_pred= predicted_centerness[positive_mask]
    positive_target= target_centerness[positive_mask]
    loss = F.binary_cross_entropy_with_logits(positive_pred, positive_target, reduction="none")

    return loss.sum() / max(normalizer, 1.0)



def compute_detection_loss(
    predictions: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> dict[str, torch.Tensor]:
    #把targets中的labels、boxes、centerness和positive_mask提取出來，並堆疊成一個batch的形式
    # targets = [
    #     target_image_1,
    #     target_image_2,
    # ]
    #target_image_1
    # ├── labels             [L]
    # ├── box_targets        [L,4]
    # ├── centerness_targets [L]
    # └── positive_mask      [L]

    # target_image_2
    # ├── labels             [L]
    # ├── box_targets        [L,4]
    # ├── centerness_targets [L]
    # └── positive_mask      [L]
    #ex target[0]["labels"] = [1, 2, 3], target[1]["labels"] = [4, 5, 6] 
    # → labels = [[1, 2, 3], [4, 5, 6]]
    labels = torch.stack([target_image["labels"] for target_image in targets], dim=0)
    box_targets = torch.stack([target_image["box_targets"] for target_image in targets], dim=0)
    centerness_targets = torch.stack([target_image["centerness_targets"] for target_image in targets], dim=0)
    positive_mask = torch.stack([target_image["positive_mask"] for target_image in targets], dim=0)
    
    #取得預測值的device和dtype，確保後續計算時不會出現裝置或數據類型不匹配的問題
    device = predictions["class_logits"].device
    dtype = predictions["class_logits"].dtype
    #將labels、box_targets、centerness_targets和positive_mask移動到與預測值相同的裝置上，並將box_targets和centerness_targets轉換為與預測值相同的數據類型
    labels = labels.to(device)
    box_targets = box_targets.to(device=device,dtype=predictions["box_regression"].dtype)
    centerness_targets = centerness_targets.to(device=device,dtype=predictions["centerness_logits"].dtype)
    positive_mask = positive_mask.to(device)
    #將原始答案labels轉換為one-hot格式，方便計算分類損失
    class_targets = labels_to_one_hot(labels, num_classes=num_classes).to(dtype)
    #計算多少個正樣本，並將normalizer設置為正樣本數量與1之間的最大值，避免除以0的情況
    num_positive = int(positive_mask.sum().item())
    normalizer = max(num_positive, 1)
    #計算classification和ltrb和centerness損失
    classification_loss = sigmoid_focal_loss(
        predictions["class_logits"],
        class_targets,
        alpha=alpha,
        gamma=gamma,
        normalizer=normalizer,
    )

    box_loss = box_regression_loss(
        predicted_ltrb=predictions["box_regression"],
        target_ltrb=box_targets,
        locations=predictions["locations"],
        positive_mask=positive_mask,
        normalizer=normalizer,
    )

    center_loss = centerness_loss(
        predictions["centerness_logits"],
        centerness_targets,
        positive_mask,
        normalizer=normalizer,
    )
    Total_loss = classification_loss + box_loss + center_loss
    return {
        "classification_loss": classification_loss,
        "box_regression_loss": box_loss,
        "centerness_loss": center_loss,
        "loss": Total_loss
    }