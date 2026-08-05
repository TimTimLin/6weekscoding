import numpy as np
from final_project_3.src.homography import build_dlt_matrix, estimate_homography_dlt,normalize_points, estimate_homography_dlt_normalized, compute_reprojection_errors

def test_build_dlt_matrix():
    src_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    dst_points = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
    A = build_dlt_matrix(src_points, dst_points)
    assert A.shape == (8, 9)
    assert np.allclose(A[0], [-0, -0, -1, 0, 0, 0, 0, 0, 0])
    assert np.allclose(A[1], [0, 0, 0, -0, -0, -1, 0, 0, 0])
    assert np.allclose(A[2], [-1, 0, -1, 0, 0, 0, 2, 0, 2])
    assert np.allclose(A[5], [0, 0, 0, 0, -1, -1, 0, 2, 2])
    assert np.allclose(A[7], [0, 0, 0, -1, -1, -1, 2, 2, 2])
    assert A.dtype == np.float64

def test_normalize_points():
    points = np.array([
        [0, 0],
        [2, 0],
        [0, 2],
        [2, 2],
    ], dtype=np.float64)

    normalized_points, T = normalize_points(points)

    expected_points = np.array([
        [-1, -1],
        [1, -1],
        [-1, 1],
        [1, 1],
    ], dtype=np.float64)

    expected_T = np.array([
        [1, 0, -1],
        [0, 1, -1],
        [0, 0, 1],
    ], dtype=np.float64)

    assert normalized_points.shape == (4, 2)
    assert T.shape == (3, 3)

    assert np.allclose(
        normalized_points,
        expected_points,
    )

    assert np.allclose(T, expected_T)

    centroid = normalized_points.mean(axis=0)
    distances = np.linalg.norm(
        normalized_points - centroid,
        axis=1,
    )

    assert np.allclose(centroid, [0, 0])
    assert np.allclose(
        distances.mean(),
        np.sqrt(2),
    )

def test_estimate_homography_dlt():
    src_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    dst_points = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
    H = estimate_homography_dlt(src_points, dst_points)
    expected_H = np.array([[2, 0, 0],
                            [0, 2, 0],
                            [0, 0, 1]])
    assert np.allclose(H, expected_H, atol=1e-6)
    assert H.shape == (3, 3)
    assert H.dtype == np.float64

def test_estimate_homography_dlt_normalized():
    src_points = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ], dtype=np.float64)

    dst_points = np.array([
        [0, 0],
        [2, 0],
        [0, 2],
        [2, 2],
    ], dtype=np.float64)

    H = estimate_homography_dlt_normalized(
        src_points,
        dst_points,
    )

    expected_H = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    assert H.shape == (3, 3)
    assert H.dtype == np.float64
    assert np.allclose(H, expected_H, atol=1e-6)

    errors = compute_reprojection_errors(
        src_points,
        dst_points,
        H,
    )

    assert np.allclose(errors, 0, atol=1e-6)   
