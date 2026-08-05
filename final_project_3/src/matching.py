
import cv2
import numpy as np
def match_features(descriptors1, descriptors2, threshold=0.75):
    #匹配兩張image的特徵點的相似度
    if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
        return []
    # Use OpenCV's BFMatcher 把個特徵點之間的距離計算出來
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    # 回傳每個特徵點的前兩個最佳匹配(image1的特徵點與哪兩個image2的特徵點最相似)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for pair in knn_matches:
        # 如果只有一個最佳匹配，則跳過該特徵點，無須進行比對
        if len(pair) < 2:
            continue
        best_match, second_best_match = pair
        #當最佳匹配的距离小于阈值乘以次佳匹配的距离时，认为是好的匹配
        if best_match.distance < threshold * second_best_match.distance:
            good_matches.append(best_match)
    return good_matches

def extract_matched_points(keypoints1, keypoints2, matches):
    #如果沒有匹配的特徵點，則回傳空的陣列，格式仍保持為[[x1, y1], [x2, y2], ...] 只是row為0
    if len(matches) == 0:
        empty_points_a = np.empty((0, 2), dtype=np.float64)
        empty_points_b = np.empty((0, 2), dtype=np.float64)
        return empty_points_a, empty_points_b
    #從匹配的特徵點中提取出對應的座標
    points_a = []
    points_b = []
    for match in matches:
        #從image1的特徵點中取出對應的座標放到points_a[list]中
        points_a.append(keypoints1[match.queryIdx].pt)
        #從image2的特徵點中取出對應的座標放到points_b[list]中
        points_b.append(keypoints2[match.trainIdx].pt)
    #將列表轉換為numpy陣列，shape為(N, 2)，其中N是匹配的特徵點數量，2表示每個點的x和y座標
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
        
    return points_a, points_b