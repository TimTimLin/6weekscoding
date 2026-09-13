import torch
from final_project_4.src.boxes import box_iou
def match_single_class_predictions(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    class_index: int,
    iou_threshold: float = 0.5,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    # 檢查「模型預測了幾張圖片」跟「GT 有幾張圖片」是不是一樣多。
    if len(predictions) != len(targets):
        raise ValueError(
            "predictions and targets must contain the same number of images."
        )
    if len(predictions) == 0:
        raise ValueError("predictions and targets must not be empty.")
    #要計算某個class的AP，先把所有圖片的預測和GT中屬於該class的框找出來
    all_scores = []
    all_true_positives = []
    all_false_positives = []
    all_ignored = []
    num_ground_truths = 0
    # 針對每一張圖片，找出屬於class_index的預測框和GT框，並計算TP、FP、ignored
    for prediction, target in zip(predictions, targets):
        # 找出prediction中屬於class_index的預測框
        prediction_class_mask = prediction["labels"] == class_index
        prediction_boxes = prediction["boxes"][prediction_class_mask]
        prediction_scores = prediction["scores"][prediction_class_mask]
        # 找出target中屬於class_index的GT框
        target_class_mask = target["labels"] == class_index
        target_boxes = target["boxes"][target_class_mask]
        target_difficult = target["difficult"][target_class_mask]       
        # 計算該class的GT框數量，忽略difficult的
        num_ground_truths += (~target_difficult).sum().item()
        image_true_positives = torch.zeros_like(
            prediction_scores,
            dtype=torch.bool,
        )
        image_false_positives = torch.zeros_like(
            prediction_scores,
            dtype=torch.bool,
        )
        image_ignored = torch.zeros_like(
            prediction_scores,
            dtype=torch.bool,
        )
        # 按照scores排序，從高到低  
        order = torch.argsort(
            prediction_scores,
            descending=True,
        )
        # 根據排序結果重新排列prediction的scores和boxes
        prediction_scores = prediction_scores[order]
        prediction_boxes = prediction_boxes[order]
        # 將target的boxes和difficult轉換到prediction的device和dtype
        target_boxes = target_boxes.to(
            device=prediction_boxes.device,
            dtype=prediction_boxes.dtype,
        )
        target_difficult = target_difficult.to(
            device=prediction_boxes.device,
        )
        #記錄每一個GT框是否已經被匹配過，初始化為False，有匹配過的GT框就不再被pred匹配
        target_already_matched = torch.zeros(
            target_boxes.shape[0],
            dtype=torch.bool,
            device=prediction_boxes.device,
        )
        if target_boxes.shape[0]== 0:
            # 如果該圖片沒有GT框，則所有預測都是FP
            image_false_positives[:] = True
        else:
            #計算每個預測框與所有GT框的IoU
            ious = box_iou(prediction_boxes, target_boxes)
            # 根據IoU匹配預測和GT
            for prediction_index in range(prediction_boxes.shape[0]):
                # 找出與該預測框IoU最大的GT框，且紀錄其IoU值和索引
                max_iou, target_index = ious[prediction_index].max(dim=0)
                if max_iou < iou_threshold:
                    image_false_positives[prediction_index] = True
                elif target_difficult[target_index]:
                        # 如果該GT框是difficult且iou大於閾值的，則標記為ignored
                        image_ignored[prediction_index] = True
                elif not target_already_matched[target_index]:
                    # 如果該GT框還沒有被匹配過，則標記為TP
                    image_true_positives[prediction_index] = True
                    target_already_matched[target_index] = True
                else:
                    # 如果該GT框已經被匹配過，則標記為FP
                    image_false_positives[prediction_index] = True
        # 將該圖片的預測結果加入到總結果中
        # ex all_scores = [
        #    tensor([0.95, 0.70]),  # image 0
        #    tensor([0.88]),        # image 1
        #    tensor([0.60, 0.40]),  # image 2
        #]
        all_scores.append(prediction_scores)
        all_true_positives.append(image_true_positives)
        all_false_positives.append(image_false_positives)
        all_ignored.append(image_ignored)
    # 合併所有圖片的預測結果 
    # ex all_scores = tensor([0.95, 0.70, 0.88, 0.60, 0.40])
    scores = torch.cat(all_scores, dim=0)
    true_positives = torch.cat(all_true_positives, dim=0)
    false_positives = torch.cat(all_false_positives, dim=0)
    ignored = torch.cat(all_ignored, dim=0)

    return (
            scores,
            true_positives,
            false_positives,
            ignored,
            int(num_ground_truths),
        )


def compute_average_precision(
    scores: torch.Tensor,
    true_positives: torch.Tensor,
    false_positives: torch.Tensor,
    num_ground_truths: int,
    ) -> torch.Tensor:
    if num_ground_truths <= 0:
        return scores.new_tensor(float("nan"))
    #依照scores大到小排序 [0.9, 0.8, 0.7...]
    _, order = torch.sort(scores, descending=True)
    #把true_positives和false_positives依照scores的ORDER排序
    # 將true_positives和false_positives轉成float型態，並計算累積和
    tp = true_positives[order].to(dtype=scores.dtype)
    fp = false_positives[order].to(dtype=scores.dtype)
    # ex TP = [0, 1]    累積 TP = [0, 1]
    #    FP = [1, 0]    累積 FP = [1, 1]
    tp = torch.cumsum(tp, dim=0)
    fp = torch.cumsum(fp, dim=0)
    # 計算precision(預測為正確的數量/所有被保留可被預測的數量)recall(預測正確的數量/實際的總數)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / num_ground_truths
    # 加入端點，讓recall從0開始到1結束，precision從0開始到0結束
    # 原本：
    #recall    = [0.0, 1.0]
    #precision = [0.0, 0.5]

    #加入端點：
    #recall    = [0.0, 0.0, 1.0, 1.0]
    #precision = [0.0, 0.0, 0.5, 0.0]
    recall = torch.cat([scores.new_tensor([0.0]), recall, scores.new_tensor([1.0])])
    precision = torch.cat([scores.new_tensor([0.0]), precision, scores.new_tensor([0.0])])
    # 從右到左，將precision的值更新為當前值與右邊值的maximum [0.0, 0.0, 0.5, 0.0] -> [0.5, 0.5, 0.5, 0.0]
    for i in range(precision.size(0) - 1, 0, -1):
        precision[i - 1] = torch.maximum(precision[i - 1], precision[i])
    # 計算AP，使用插值法計算precision-recall曲線下的面積
    #recall    = [0.0, 0.0, 1.0, 1.0]
    #precision = [0.0, 0.0, 0.5, 0.0]

    #AP = (1.0 - 0.0) × 0.5
    #   = 0.5
    ap = torch.sum((recall[1:] - recall[:-1]) * precision[1:])
    return ap

def compute_mean_average_precision(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    if num_classes <= 0:    
        raise ValueError("num_classes must be greater than 0.")
    per_class_ap = []
    per_class_num_ground_truths = []
    for class_index in range(num_classes):
        # 針對每個class，計算該class的AP
        (
            scores,
            true_positives,
            false_positives,
            ignored,            #iou大於閾值且GT是difficult的預測框會被標記為ignored
            num_ground_truths,  #非difficult的GT數量
        ) = match_single_class_predictions(
            predictions=predictions,
            targets=targets,
            class_index=class_index,
            iou_threshold=iou_threshold,
        )
        # 把被標記為ignored的預測框排除掉，因為這些預測框不會影響AP的計算
        valid_predictions_mask = ~ignored
        # 只保留有效的預測框
        scores = scores[valid_predictions_mask]
        true_positives = true_positives[valid_predictions_mask]
        false_positives = false_positives[valid_predictions_mask]

        ap = compute_average_precision(
            scores=scores,
            true_positives=true_positives,
            false_positives=false_positives,
            num_ground_truths=num_ground_truths,
        )
        # 把每個class的AP和該class的GT數量加入到列表中
        per_class_ap.append(ap)
        per_class_num_ground_truths.append(num_ground_truths)
    # 把每個class的AP和GT數量轉換成tensor，方便後續計算
    ap_per_class = torch.stack(per_class_ap)
    num_ground_truths_per_class = torch.tensor(per_class_num_ground_truths, dtype=torch.int64, device=ap_per_class.device)
    # 計算mAP，只考慮有GT的class，忽略沒有GT的class
    valid_classes_mask = num_ground_truths_per_class > 0
    if valid_classes_mask.any():
        mean_ap = ap_per_class[valid_classes_mask].mean()
    else:
        mean_ap = ap_per_class.new_tensor(float("nan"))
    return {
        "map": mean_ap,
        "ap_per_class": ap_per_class,
        "num_ground_truths_per_class": num_ground_truths_per_class,
        }