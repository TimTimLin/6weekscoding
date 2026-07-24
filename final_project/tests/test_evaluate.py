
import torch

from final_project.src.evaluate import (
    build_confusion_matrix,
    classwise_accuracy,
    get_top_confused_classes,
)

def test_confusion_matrix_counts_duplicates():
    #檢查是否正確計算了混淆矩陣中的重複項
    true_labels = torch.tensor([0, 0, 0, 1])
    predictions = torch.tensor([1, 1, 1, 2])
    # num_classes=3，因為我們有三個類別：0, 1, 2
    confusion = build_confusion_matrix(
        true_labels,
        predictions,
        num_classes=3,
    )
    # label 0 被預測為 label 1 的次數應該是3
    # label 1 被預測為 label 2 的次數應該是1
    assert confusion[0, 1].item() == 3
    assert confusion[1, 2].item() == 1
    # 確認混淆矩陣的總和是否等於樣本數
    assert confusion.sum().item() == 4
    
def test_classwise_accuracy_and_top_confusion():
    confusion = torch.tensor([
        [7, 2, 1],
        [0, 9, 1],
        [2, 0, 8],
    ])

    accuracy = classwise_accuracy(confusion)
    # 準確率計算公式：accuracy = correct_predictions / total_per_class
    # 對於類別0: accuracy = 7 / (7 + 2 + 1) = 7 / 10 = 0.7
    # 對於類別1: accuracy = 9 / (0 + 9 + 1) = 9 / 10 = 0.9
    # 對於類別2: accuracy = 8 / (2 + 0 + 8) = 8 / 10 = 0.8
    assert torch.allclose(
        accuracy,
        torch.tensor([0.7, 0.9, 0.8]),
    )
    # 測試 get_top_confused_classes 函數
    values, indices = get_top_confused_classes(
        confusion,
        class_index=0,
        k=2,
    )
    # 對於類別0，混淆矩陣的第一行是 [7, 2, 1]，我們將正確預測的數量設為0，所以變成 [0, 2, 1]。
    # 前兩個最大的值是2和1，對應的索引是1和2。
    assert values.tolist() == [2, 1]
    assert indices.tolist() == [1, 2]
