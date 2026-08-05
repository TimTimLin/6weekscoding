from types import SimpleNamespace

import numpy as np

from final_project_3.src.matching import (
    extract_matched_points,
    match_features,
)


def test_match_features_with_none_descriptors():
    matches = match_features(None, None)

    assert matches == []


def test_match_features_lowe_ratio_accepts_clear_match():
    descriptors1 = np.zeros((1, 128), dtype=np.float32)

    descriptors2 = np.array([
        np.zeros(128),
        np.full(128, 10.0),
    ], dtype=np.float32)

    matches = match_features(
        descriptors1,
        descriptors2,
        threshold=0.75,
    )

    assert len(matches) == 1
    assert matches[0].queryIdx == 0
    assert matches[0].trainIdx == 0


def test_match_features_lowe_ratio_rejects_ambiguous_match():
    descriptors1 = np.zeros((1, 128), dtype=np.float32)

    descriptors2 = np.array([
        np.full(128, 0.10),
        np.full(128, 0.11),
    ], dtype=np.float32)

    matches = match_features(
        descriptors1,
        descriptors2,
        threshold=0.75,
    )

    assert matches == []


def test_extract_matched_points():
    keypoints1 = [
        SimpleNamespace(pt=(10.0, 20.0)),
        SimpleNamespace(pt=(30.0, 40.0)),
    ]

    keypoints2 = [
        SimpleNamespace(pt=(100.0, 200.0)),
    ]

    matches = [
        SimpleNamespace(
            queryIdx=1,
            trainIdx=0,
        )
    ]

    points1, points2 = extract_matched_points(
        keypoints1,
        keypoints2,
        matches,
    )

    assert points1.shape == (1, 2)
    assert points2.shape == (1, 2)
    assert points1.dtype == np.float64
    assert points2.dtype == np.float64

    assert np.allclose(points1, [[30.0, 40.0]])
    assert np.allclose(points2, [[100.0, 200.0]])