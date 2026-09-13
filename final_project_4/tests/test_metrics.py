import torch

from final_project_4.src.metrics import compute_average_precision, compute_mean_average_precision, match_single_class_predictions


def assert_ap_close(actual: torch.Tensor, expected: float) -> None:
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == torch.Size([])

    expected_tensor = actual.new_tensor(expected)
    torch.testing.assert_close(actual, expected_tensor)


def test_ap_perfect_predictions():
    ap = compute_average_precision(
        scores=torch.tensor([0.9, 0.8, 0.7]),
        true_positives=torch.tensor([1, 1, 1]),
        false_positives=torch.tensor([0, 0, 0]),
        num_ground_truths=3,
    )

    assert_ap_close(ap, 1.0)


def test_ap_sorts_by_score():
    # 原始順序：
    # score=0.8 → TP
    # score=0.9 → FP
    #
    # 排序後：
    # score=0.9 → FP
    # score=0.8 → TP
    ap = compute_average_precision(
        scores=torch.tensor([0.8, 0.9]),
        true_positives=torch.tensor([1, 0]),
        false_positives=torch.tensor([0, 1]),
        num_ground_truths=1,
    )

    assert_ap_close(ap, 0.5)


def test_ap_partial_recall():
    ap = compute_average_precision(
        scores=torch.tensor([0.9]),
        true_positives=torch.tensor([1]),
        false_positives=torch.tensor([0]),
        num_ground_truths=2,
    )

    # 只找到 2 個 GT 中的 1 個
    assert_ap_close(ap, 0.5)


def test_ap_no_predictions():
    ap = compute_average_precision(
        scores=torch.empty(0),
        true_positives=torch.empty(0, dtype=torch.bool),
        false_positives=torch.empty(0, dtype=torch.bool),
        num_ground_truths=2,
    )

    assert_ap_close(ap, 0.0)


def test_ap_no_ground_truths():
    ap = compute_average_precision(
        scores=torch.tensor([0.9]),
        true_positives=torch.tensor([0]),
        false_positives=torch.tensor([1]),
        num_ground_truths=0,
    )

    assert torch.isnan(ap).item()
def test_match_single_class_predictions():
    predictions = [
        {
            # image 0：刻意把 0.8 放在 0.9 前面，測試函式是否排序
            "boxes": torch.tensor([
                [0.0, 0.0, 10.0, 10.0],    # duplicate FP
                [0.0, 0.0, 10.0, 10.0],    # TP
                [20.0, 20.0, 30.0, 30.0],  # difficult → ignored
                [40.0, 40.0, 50.0, 50.0],  # no match → FP
            ]),
            "labels": torch.tensor(
                [1, 1, 1, 1],
                dtype=torch.int64,
            ),
            "scores": torch.tensor([
                0.8,
                0.9,
                0.7,
                0.6,
            ]),
        },
        {
            # image 1：有 class 1 prediction，但沒有 class 1 GT，只有 class 2 GT
            "boxes": torch.tensor([
                [60.0, 60.0, 70.0, 70.0],
            ]),
            "labels": torch.tensor(
                [1],
                dtype=torch.int64,
            ),
            "scores": torch.tensor([0.5]),
        },
    ]

    targets = [
        {
            "boxes": torch.tensor([
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
            ]),
            "labels": torch.tensor(
                [1, 1],
                dtype=torch.int64,
            ),
            "difficult": torch.tensor([
                False,
                True,
            ]),
        },
        {
            # 這張圖只有 class 2 GT
            "boxes": torch.tensor([
                [60.0, 60.0, 70.0, 70.0],
            ]),
            "labels": torch.tensor(
                [2],
                dtype=torch.int64,
            ),
            "difficult": torch.tensor([
                False,
            ]),
        },
    ]

    scores,true_positives,false_positives,ignored,num_ground_truths= match_single_class_predictions(
        predictions=predictions,
        targets=targets,
        class_index=1,
        iou_threshold=0.5,
    )

    torch.testing.assert_close(
        scores,
        torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]),
    )
    torch.testing.assert_close(
        true_positives,
        torch.tensor([True, False, False, False, False]),
    )
    torch.testing.assert_close(
        false_positives,
        torch.tensor([False, True, False, True, True]),
    )
    torch.testing.assert_close(
        ignored,
        torch.tensor([False, False, True, False, False]),
    )
    assert num_ground_truths == 1
    
def test_compute_mean_average_precision():
    predictions = [
        {
            "boxes": torch.tensor([
                [40.0, 40.0, 50.0, 50.0],  # class 1 difficult
                [0.0, 0.0, 10.0, 10.0],    # class 0 TP
                [60.0, 60.0, 70.0, 70.0],  # class 1 FP
                [20.0, 20.0, 30.0, 30.0],  # class 1 TP
            ]),
            "labels": torch.tensor(
                [1, 0, 1, 1],
                dtype=torch.int64,
            ),
            "scores": torch.tensor([
                0.95,
                0.90,
                0.90,
                0.80,
            ]),
        },
    ]

    targets = [
        {
            "boxes": torch.tensor([
                [0.0, 0.0, 10.0, 10.0],    # class 0 valid
                [20.0, 20.0, 30.0, 30.0],  # class 1 valid
                [40.0, 40.0, 50.0, 50.0],  # class 1 difficult
            ]),
            "labels": torch.tensor(
                [0, 1, 1],
                dtype=torch.int64,
            ),
            "difficult": torch.tensor([
                False,
                False,
                True,
            ]),
        },
    ]

    metrics = compute_mean_average_precision(
        predictions=predictions,
        targets=targets,
        num_classes=2,
        iou_threshold=0.5,
    )

    torch.testing.assert_close(
        metrics["ap_per_class"],
        torch.tensor([1.0, 0.5]),
    )
    torch.testing.assert_close(
        metrics["map"],
        torch.tensor(0.75),
    )
    torch.testing.assert_close(
        metrics["num_ground_truths_per_class"],
        torch.tensor([1, 1], dtype=torch.int64),
    )