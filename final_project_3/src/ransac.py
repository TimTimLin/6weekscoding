import numpy as np
from final_project_3.src.homography import estimate_homography_dlt, compute_reprojection_errors
def compute_inlier_mask(errors, threshold):
    errors = np.asarray(errors, dtype=np.float64)
    inliers = errors <= threshold
    return inliers
def ransac_homography(src_points, dst_points, num_iterations=1000, threshold=5.0):
    """RANSAC algorithm for estimating homography between two sets of points.

    全部 N 組 correspondences
        │
        ├─ 隨機抽 4 組
        │      │
        │      └─ DLT → 候選 H
        │
        ├─ 候選 H 投影全部 N 組
        │      │
        │      ├─ 計算 N 個 reprojection errors
        │      ├─ 產生 N 個 inlier/outlier mask
        │      └─ 計算 inlier count
        │
        └─ 重複多次，保留 inlier count 最大的 H """
    best_H = None
    best_inlier_mask = None
    max_inliers = 0
    #不足4個點或兩個點數量不一致，無法計算單應性矩陣
    if src_points.shape[0] < 4 or src_points.shape[0] != dst_points.shape[0]:
        return None, None

    for _ in range(num_iterations):
        # 隨機選擇4個對應點
        indices = np.random.choice(src_points.shape[0], 4, replace=False)
        
        src_subset = src_points[indices]
        dst_subset = dst_points[indices]

        # 使用DLT估計單應性矩陣
        H = estimate_homography_dlt(src_subset, dst_subset)

        # 計算對N個點的重投影誤差
        errors = compute_reprojection_errors(src_points, dst_points, H)

        # 計算這N個點的重投影誤差是否小於閾值的inlier mask
        inlier_mask = compute_inlier_mask(errors, threshold)
        # 計算inlier的數量
        num_inliers = np.sum(inlier_mask)

        # 更新最佳模型
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inlier_mask = inlier_mask
    # 剛剛的best_H和best_inlier_mask只針對4個點得到的，最後使用最佳的inlier mask重新估計適用於N個點的單應性矩陣
    if best_inlier_mask is None or max_inliers < 4:
        return None, None
    final_H = estimate_homography_dlt(src_points[best_inlier_mask], dst_points[best_inlier_mask])
    final_errors = compute_reprojection_errors(src_points, dst_points, final_H)
    final_inlier_mask = compute_inlier_mask(final_errors, threshold)
    return final_H, final_inlier_mask
