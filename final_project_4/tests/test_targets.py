
import torch
from final_project_4.src.targets import (
    flatten_predictions, generate_locations, 
    compute_ltrb_targets, compute_inside_box_mask,
    compute_centerness_targets,
    match_locations_to_boxes,compute_regression_range_mask,
    assign_targets_to_locations
)
def test_flatten_predictions_shapes():
    predictions = {
        "p3": {
            "class_logits": torch.randn(2, 20, 32, 48),
             "box_regression": torch.randn(2, 4, 32, 48),
            "centerness_logits": torch.randn(2, 1, 32, 48),
        },
        "p4": {
            "class_logits": torch.randn(2, 20, 16, 24),
            "box_regression": torch.randn(2, 4, 16, 24),
            "centerness_logits": torch.randn(2, 1, 16, 24),
        },
        "p5": {
            "class_logits": torch.randn(2, 20, 8, 12),
            "box_regression": torch.randn(2, 4, 8, 12),
            "centerness_logits": torch.randn(2, 1, 8, 12),
        },
    }

    flattened = flatten_predictions(predictions)

    assert flattened["class_logits"].shape == (2, (32 * 48 + 16 * 24 + 8 * 12), 20)
    assert flattened["box_regression"].shape == (2, (32 * 48 + 16 * 24 + 8 * 12), 4)
    assert flattened["centerness_logits"].shape == (2, (32 * 48 + 16 * 24 + 8 * 12))
    
def test_generate_locations_shapes():
    predictions = {
        "p3": {
            "class_logits": torch.randn(2, 20, 32, 48),
        },
        "p4": {
            "class_logits": torch.randn(2, 20, 16, 24),
        },
        "p5": {
            "class_logits": torch.randn(2, 20, 8, 12),
        },
    }

    stride = {"p3": 8, "p4": 16, "p5": 32}

    locations, strides = generate_locations(predictions, stride)

    assert locations.shape == (32 * 48 + 16 * 24 + 8 * 12, 2)
    assert strides.shape == (32 * 48 + 16 * 24 + 8 * 12,)

    # P3 的第一個 location
    torch.testing.assert_close(
        locations[0],
        torch.tensor([4.0, 4.0]),
    )

    # P3：確認 x 在同一個 row 中先增加
    torch.testing.assert_close(
        locations[1],
        torch.tensor([12.0, 4.0]),
    )

    # P3：確認跑完一整列後，才換到下一個 y
    torch.testing.assert_close(
        locations[48],
        torch.tensor([4.0, 12.0]),
    )

    # P4 的第一個 location 
    torch.testing.assert_close(
        locations[1536],
        torch.tensor([8.0, 8.0]),
    )

    # P5 的第一個 location
    torch.testing.assert_close(
        locations[1920],
        torch.tensor([16.0, 16.0]),
    )

    assert torch.all(strides[:1536] == 8)
    assert torch.all(strides[1536:1920] == 16)
    assert torch.all(strides[1920:] == 32)
    
def test_compute_ltrb_targets_shapes():
    locations = torch.tensor([
        [4.0, 4.0],
        [20.0, 4.0],
    ])

    boxes = torch.tensor([
        [0.0, 0.0, 16.0, 12.0],
        [8.0, 0.0, 24.0, 16.0]
    ])

    ltrb_targets = compute_ltrb_targets(locations, boxes)

    assert ltrb_targets.shape == (2, 2, 4)
    torch.testing.assert_close(ltrb_targets[0][0], torch.tensor([4.0, 4.0, 12.0, 8.0]))
    torch.testing.assert_close(ltrb_targets[1][0], torch.tensor([20.0, 4.0, -4.0, 8.0]))
    torch.testing.assert_close(ltrb_targets[0][1], torch.tensor([-4.0, 4.0, 20.0, 12.0]))
    torch.testing.assert_close(ltrb_targets[1][1], torch.tensor([12.0, 4.0, 4.0, 12.0]))
    
def test_compute_inside_box_mask():
    ltrb_targets = torch.tensor([
        [
            [4.0, 4.0, 12.0, 8.0],
            [-4.0, 4.0, 20.0, 12.0],
        ],
        [
            [20.0, 4.0, -4.0, 8.0],
            [12.0, 4.0, 4.0, 12.0],
        ],
    ])

    expected = torch.tensor([
        [True, False],
        [False, True],
    ])

    inside_mask = compute_inside_box_mask(ltrb_targets)

    assert inside_mask.shape == (2, 2)
    assert inside_mask.dtype == torch.bool
    torch.testing.assert_close(inside_mask, expected)
    
def test_match_locations_to_boxes():
    inside_mask = torch.tensor([
        [True, True],
        [True, False],
        [False, False],
    ])

    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],  # area 100
        [2.0, 2.0, 8.0, 8.0],     # area 36
    ])

    matched_indices, positive_mask = match_locations_to_boxes(
        inside_mask,
        boxes,
    )

    expected_indices = torch.tensor([1, 0, -1])
    expected_positive = torch.tensor([True, True, False])

    assert matched_indices.shape == (3,)
    assert matched_indices.dtype == torch.int64
    assert positive_mask.shape == (3,)
    assert positive_mask.dtype == torch.bool

    torch.testing.assert_close(matched_indices, expected_indices)
    torch.testing.assert_close(positive_mask, expected_positive)
    
#建立一個測試函數，測試當所有位置都不在任何邊界框內時，match_locations_to_boxes函數的行為
def test_match_locations_all_background():
    inside_mask = torch.zeros(
        (3, 2),
        dtype=torch.bool,
    )

    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 20.0, 30.0, 30.0],
    ])

    matched_indices, positive_mask = match_locations_to_boxes(
        inside_mask,
        boxes,
    )

    torch.testing.assert_close(
        matched_indices,
        torch.tensor([-1, -1, -1]),
    )

    torch.testing.assert_close(
        positive_mask,
        torch.tensor([False, False, False]),
    )

# 建立一個測試函數，測試當沒有邊界框時，match_locations_to_boxes函數的行為
def test_match_locations_to_boxes_empty_boxes():
    inside_mask = torch.zeros(
        (3, 0),
        dtype=torch.bool,
    )

    boxes = torch.empty(
        (0, 4),
        dtype=torch.float32,
    )

    matched_indices, positive_mask = match_locations_to_boxes(
        inside_mask,
        boxes,
    )

    torch.testing.assert_close(
        matched_indices,
        torch.tensor([-1, -1, -1]),
    )

    torch.testing.assert_close(
        positive_mask,
        torch.tensor([False, False, False]),
    )
    
def test_compute_regression_range_mask():
    ltrb_targets = torch.tensor([
        [[10.0, 10.0, 10.0, 10.0]],   # max=10,   P3 valid
        [[64.0, 20.0, 20.0, 20.0]],   # max=64,   P3 boundary valid
        [[80.0, 20.0, 20.0, 20.0]],   # max=80,   P3 invalid
        [[80.0, 20.0, 20.0, 20.0]],   # max=80,   P4 valid
        [[128.0, 20.0, 20.0, 20.0]],  # max=128,  P4 boundary valid
        [[140.0, 20.0, 20.0, 20.0]],  # max=140,  P4 invalid
        [[140.0, 20.0, 20.0, 20.0]],  # max=140,  P5 valid
    ])

    location_strides = torch.tensor([
        8.0,   # P3
        8.0,   # P3
        8.0,   # P3
        16.0,  # P4
        16.0,  # P4
        16.0,  # P4
        32.0,  # P5
    ])

    regression_ranges = {
        8: (0.0, 64.0),
        16: (64.0, 128.0),
        32: (128.0, float("inf")),
    }

    range_mask = compute_regression_range_mask(
        ltrb_targets,
        location_strides,
        regression_ranges,
    )

    expected = torch.tensor([
        [True],   # 10  → P3
        [True],   # 64  → P3 boundary
        [False],  # 80  → P3 invalid
        [True],   # 80  → P4
        [True],   # 128 → P4 boundary
        [False],  # 140 → P4 invalid
        [True],   # 140 → P5
    ])

    assert range_mask.shape == (7, 1)
    assert range_mask.dtype == torch.bool
    torch.testing.assert_close(range_mask, expected)
    
def test_assign_targets_to_locations():
    locations = torch.tensor([
        [4.0, 4.0],
        [8.0, 6.0],
        [20.0, 4.0],
    ])

    location_strides = torch.tensor([
        8.0,
        8.0,
        8.0,
    ])

    boxes = torch.tensor([
        [0.0, 0.0, 16.0, 12.0],
    ])

    labels = torch.tensor([11], dtype=torch.int64)

    regression_ranges = {
        8: (0.0, 64.0),
        16: (64.0, 128.0),
        32: (128.0, float("inf")),
    }

    targets = assign_targets_to_locations(
        locations,
        location_strides,
        boxes,
        labels,
        regression_ranges,
    )

    torch.testing.assert_close(
        targets["labels"],
        torch.tensor([11, 11, -1]),
    )

    torch.testing.assert_close(
        targets["box_targets"],
        torch.tensor([
            [4.0, 4.0, 12.0, 8.0],
            [8.0, 6.0, 8.0, 6.0],
            [0.0, 0.0, 0.0, 0.0],
        ]),
    )

    torch.testing.assert_close(
        targets["positive_mask"],
        torch.tensor([True, True, False]),
    )
    
    torch.testing.assert_close(
        targets["matched_gt_indices"],
        torch.tensor([0, 0, -1]),
    )
    
    assert targets["labels"].dtype == torch.int64
    assert targets["box_targets"].dtype == torch.float32
    assert targets["centerness_targets"].dtype == torch.float32
    assert targets["labels"].shape == (3,)
    assert targets["box_targets"].shape == (3, 4)
    assert targets["centerness_targets"].shape == (3,)

    
def test_compute_centerness_targets():
    box_targets = torch.tensor([
        [4.0, 4.0, 12.0, 8.0],
        [8.0, 6.0, 8.0, 6.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    positive_mask = torch.tensor([True, True, False])

    centerness = compute_centerness_targets(
        box_targets,
        positive_mask,
    )

    expected = torch.tensor([
        (1.0 / 6.0) ** 0.5,
        1.0,
        0.0,
    ])

    torch.testing.assert_close(centerness, expected)