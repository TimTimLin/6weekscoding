import numpy as np


# basic DLT
####################################################################################
def build_dlt_matrix(src_points: np.ndarray,dst_points: np.ndarray,) -> np.ndarray:
    #一組 correspondence(src(x,y)->dst(u,v)) ⇒ 兩列(row)
    #有多少組correspondence
    num_points = src_points.shape[0]
    #建立A的空陣列
    A = np.zeros((2 * num_points, 9))

    for i in range(num_points):
        x, y = src_points[i]
        u, v = dst_points[i]

        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]

    return A

def estimate_homography_dlt(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    A = build_dlt_matrix(src_points, dst_points)
    #使用SVD分解來求解H
    _, _, Vt = np.linalg.svd(A)
    #SVD的最後一列對應於最小奇異值，這就是我們需要的H向量
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]  # Normalize so that H[2, 2] is 1


## Normalization dLT
################################################################################

def normalize_points(points: np.ndarray):
    #計算質心 axis=0表示對每一col計算平均值，(x1, y1),(x2, y2),...得到x和y的平均值
    centroid = np.mean(points, axis=0)    
    #計算每個點到質心相差的座標
    shifted_points = points - centroid
    #計算每個點到質心的距離 axis=1表示對每一row(x,y)計算平方和再開根號，得到每個點到質心的距離
    distances = np.linalg.norm(shifted_points, axis=1)
    #計算平均距離
    mean_distance = np.mean(distances)
    #計算縮放因子，使得平均距離為 sqrt(2) why sqrt(2)? 因為我們希望將點雲縮放到x, y的範圍[-1, 1]內。
    #ex: mean_distance = 20, scale = sqrt(2)/20= 0.0707, 這樣縮放後的點雲平均距離就會變成 sqrt(2)
    scale = np.sqrt(2) / mean_distance
    #建立轉換矩陣 T 目標讓centroid移到原點，並且縮放到平均距離為 sqrt(2)
    #new_x = scale * (x - centroid_x)
    #new_y = scale * (y - centroid_y)
    #此矩陣T @ [x, y, 1]^T = [new_x, new_y, 1]^T
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    #hstack就是把初始每個點的(x, y)加上1變成(x, y, 1)的齊次座標
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    #將轉換矩陣T應用到每個點上，得到新的齊次座標 T[3x3] @ [x, y, 1]^T = [new_x, new_y, 1]^T
    normalized_homogeneous = (T @ points_homogeneous.T).T
    #(x, y, 1)取出前兩個座標(x, y)，得到新的非齊次座標
    normalized_points = normalized_homogeneous[:, :2]
    return normalized_points, T

def estimate_homography_dlt_normalized(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    原始 source point
            │
            │ T_src
            ▼
    normalized source point
            │
            │ H_normalized
            ▼
    normalized destination point
            │
            │ inv(T_dst)
            ▼
    原始 destination point
    """
    
    #對src_points和dst_points進行正規化
    normalized_src, T_src = normalize_points(src_points)
    normalized_dst, T_dst = normalize_points(dst_points)

    A = build_dlt_matrix(normalized_src, normalized_dst)
    #使用SVD分解來求解H_normalized 
    _, _, Vt = np.linalg.svd(A)
    #SVD的最後一列對應於最小奇異值，這就是我們需要的H_normalized向量
    H_normalized = Vt[-1].reshape(3, 3)
    #dst_points = inv(T_dst) @ H_normalized @ T_src @ src_points
    #將正規化後的單應性矩陣轉換回原始座標系
    H = np.linalg.inv(T_dst) @ H_normalized @ T_src
    return H / H[2, 2]  # Normalize so that H[2, 2] is 1


def compute_reprojection_errors(src_points, dst_points, H):
    #將src_points轉換為齊次座標  (x, y) -> (x, y, 1)
    src_homogeneous = np.hstack([src_points, np.ones((src_points.shape[0], 1))])
    #將H應用於src_points         H[3x3] @ [x, y, 1]^T = [x', y', z']^T
    projected_points = (H @ src_homogeneous.T).T
    #將投影點轉換回非齊次座標       x' = x/z, y' = y/z
    projected_points /= projected_points[:, 2][:, np.newaxis]
    #計算重投影誤差  pred_dst和true_dst的誤差 sqrt((x' - u)² + (y' - v)²)
    errors = np.linalg.norm(projected_points[:, :2] - dst_points, axis=1)
    return errors
