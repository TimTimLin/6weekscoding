import torch
from final_project_5.src.metrics import confusion_matrix_from_predictions, iou_from_confusion_matrix

def test_confusion_matrix_from_predictions():
    targets = torch.tensor(
        [[[0, 0, 0],
        [1, 1, 2]]],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [[[0, 1, 1],
        [1, 2, 2]]],
        dtype=torch.int64,
    )

    expected = torch.tensor(
        [[1, 2, 0],
        [0, 1, 1],
        [0, 0, 1]],
        dtype=torch.int64,
    )
    actual = confusion_matrix_from_predictions(
        predictions,
        targets,
        num_classes=3,
    )

    assert actual.shape == (3, 3)
    assert actual.dtype == torch.int64
    assert actual.sum().item() == 6
    assert torch.equal(actual, expected)
    
def test_iou_from_confusion_matrix():
    confusion_matrix = torch.tensor(
        [
            [1, 2, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=torch.int64,
    )
    per_class_iou, mean_iou = iou_from_confusion_matrix(confusion_matrix)
    expected_per_class = torch.tensor(
        [1.0 / 3.0, 1.0 / 4.0, 1.0 / 2.0],
        dtype=torch.float64,
    )

    expected_mean = torch.tensor(
        13.0 / 36.0,
        dtype=torch.float64,
    )
    assert torch.allclose(mean_iou, expected_mean)
    assert torch.allclose(per_class_iou, expected_per_class)
    assert per_class_iou.shape == (3,)
    assert mean_iou.ndim == 0
    assert per_class_iou.dtype == torch.float64
    assert mean_iou.dtype == torch.float64
    
def test_iou_ignores_classes_with_zero_union():
    confusion_matrix = torch.tensor(
        [
            [2, 1, 0],
            [0, 3, 0],
            [0, 0, 0],
        ],
        dtype=torch.int64,
    )
    per_class_iou, mean_iou = iou_from_confusion_matrix(confusion_matrix)
    
    
    
    
    assert torch.isclose(
        per_class_iou[0],
        torch.tensor(2.0 / 3.0, dtype=torch.float64),
    )

    assert torch.isclose(
        per_class_iou[1],
        torch.tensor(3.0 / 4.0, dtype=torch.float64),
    )

    assert torch.isnan(per_class_iou[2])

    assert torch.isclose(
        mean_iou,
        torch.tensor(17.0 / 24.0, dtype=torch.float64),
    )