import numpy as np

from final_project_3.src.warping import (
    transform_corners,
    warp_to_common_canvas,
)


def test_transform_corners_with_translation():
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    translation = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1],
    ], dtype=np.float64)

    corners = transform_corners(image, translation)

    expected_corners = np.array([
        [5, 3],
        [24, 3],
        [24, 12],
        [5, 12],
    ], dtype=np.float64)

    assert corners.shape == (4, 2)
    assert np.allclose(corners, expected_corners)


def test_warp_to_common_canvas():
    image1 = np.zeros((10, 20, 3), dtype=np.uint8)
    image2 = np.zeros((10, 20, 3), dtype=np.uint8)

    image1[:, :, 0] = 255  # BGR blue
    image2[:, :, 1] = 255  # BGR green

    H_image2_from_image1 = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1],
    ], dtype=np.float64)

    (warped1, mask1), (warped2, mask2) = warp_to_common_canvas(
        image1,
        image2,
        H_image2_from_image1,
    )

    assert warped1.shape == (13, 25, 3)
    assert warped2.shape == (13, 25, 3)

    assert mask1.shape == (13, 25)
    assert mask2.shape == (13, 25)

    assert mask1.dtype == np.bool_
    assert mask2.dtype == np.bool_

    assert mask1.sum() > 0
    assert mask2.sum() > 0
    assert np.count_nonzero(mask1 & mask2) > 0