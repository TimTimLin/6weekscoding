from final_project_3.src.ransac import compute_inlier_mask, ransac_homography
import numpy as np
def test_compute_inlier_mask():
    errors = np.array([0.8, 1.2, 2.1, 5.0, 0.4])

    mask = compute_inlier_mask(errors, threshold=2.0)

    assert mask.shape == (5,)
    assert mask.dtype == np.bool_
    assert np.array_equal(
        mask,
        np.array([True, True, False, False, True]),
    )
    assert mask.sum() == 3
    
def test_ransac_homography_with_outliers():
    src_points = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [2, 1],
        [0, 2],
        [2, 2],
    ], dtype=np.float64)

    true_H = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    dst_points = 2.0 * src_points

    # 人為加入兩個錯誤 match
    dst_points[-1] = [50, -20]
    dst_points[-2] = [100, 30]

    H, mask = ransac_homography(
        src_points,
        dst_points,
        num_iterations=1000,
        threshold=1.0,
    )

    assert H is not None
    assert mask.shape == (8,)
    assert mask.sum() >= 6
    assert np.allclose(H, true_H, atol=1e-5)
    
def test_ransac_homography_with_too_few_points():
    src_points = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
    ], dtype=np.float64)

    dst_points = np.array([
        [0, 0],
        [2, 0],
        [0, 2],
    ], dtype=np.float64)

    H, mask = ransac_homography(
        src_points,
        dst_points,
    )

    assert H is None
    assert mask is None