import numpy as np
import cv2
def transform_corners(image, H):
    #將圖像的四個角點轉換為新的位置 Homography Warp 將原本的正方形圖扭曲
    height, width = image.shape[:2]
    corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float64)
    #將角點轉換為齊次座標 (x, y) -> (x, y, 1)
    corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
    #H應用於角點，得到新的角點位置 H[3x3] @ [x, y, 1]^T = [x', y', z']^T
    transformed_corners = (H @ corners_homogeneous.T).T
    #將投影點轉換回非齊次座標 x' = x/z, y' = y/z
    transformed_corners /= transformed_corners[:, 2][:, np.newaxis]
    #回傳新的角點位置 (x', y',1) -> (x', y') 只取全部row的前兩個column
    return transformed_corners[:, :2] 
def compute_canvas_bounds(corners_a, corners_b):
    #計算canvas的邊界，將兩個圖像(A、B)的角點合併，找出最小和最大值
    all_corners = np.vstack([corners_a, corners_b])
    min_x = np.min(all_corners[:, 0])
    min_y = np.min(all_corners[:, 1])
    max_x = np.max(all_corners[:, 0])
    max_y = np.max(all_corners[:, 1])
    return min_x, min_y, max_x, max_y

def finalize_canvas_bounds(min_x, min_y, max_x, max_y):
    x_min = int(np.floor(min_x))
    y_min = int(np.floor(min_y))
    x_max = int(np.ceil(max_x))
    y_max = int(np.ceil(max_y))

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    #建立平移矩陣，使原本(min_x, min_y)的點移動到(0, 0)的位置，這樣canvas的左上角就是(0, 0)
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1],
    ], dtype=np.float64)

    return translation, (width, height)
def warp_image_to_canvas(image, H_canvas_from_image, canvas_size):
    #opencv的warpPerspective函數需要輸入canvas的大小(width, height)
    canvas_width, canvas_height = canvas_size
    #將image warp到canvas上，使用cv2.warpPerspective
    warped_image = cv2.warpPerspective(image, H_canvas_from_image, (canvas_width, canvas_height))
    #建立一個全白的mask，大小與image相同，方便後續將warped_image放到canvas上
    all_white_mask = np.full(image.shape[:2],255,dtype=np.uint8)
    #將warped_mask放到canvas上，協助分辨warped_image的有效區域，避免黑色背景影響後續的圖像融合
    warped_all_white_mask = cv2.warpPerspective(all_white_mask, H_canvas_from_image, (canvas_width, canvas_height),flags=cv2.INTER_NEAREST)
    #將warped_all_white_mask轉換為布林值，true代表有效區域，方便後續使用
    valid_mask = warped_all_white_mask > 0
    return warped_image, valid_mask


#合併上述的功能，將兩張圖像warp到同一個canvas上，並回傳warped_image和valid_mask
def warp_to_common_canvas(image1, image2, H_image2_from_image1):
    #為image2的H建立一個單位矩陣，表示image2不需要warp，是原本的圖像，H_image2_from_image1是將image1 warp到image2的H
    identity = np.eye(3, dtype=np.float64)
    #將image1的角點warp到image2的canvas上，並將image2的角點warp到image2的canvas上(這個完全沒平移)，計算出canvas的邊界
    corners1 = transform_corners(image1, H_image2_from_image1)
    corners2 = transform_corners(image2, identity)
    #計算canvas的邊界
    bounds = compute_canvas_bounds(corners1, corners2)
    #建立平移矩陣translation，使canvas的左上角為(0, 0)，並計算canvas的大小
    translation, canvas_size = finalize_canvas_bounds(*bounds)
    #H_image2_from_image1是先將image1 warp到image2的H，然後translation再將canvas左上角平移到(0, 0)
    #identity是先將image2保持原位，然後translation再將canvas左上角平移到(0, 0)
    warped1, mask1 = warp_image_to_canvas(
        image1,
        translation @ H_image2_from_image1,
        canvas_size,
    )

    warped2, mask2 = warp_image_to_canvas(
        image2,
        translation @ identity,
        canvas_size,
    )
    return (warped1, mask1), (warped2, mask2)