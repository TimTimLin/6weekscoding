import cv2
import numpy as np
from final_project_3.src.features import detect_sift_features

def test_detect_sift_features():
    test_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(test_image, (50, 50), 20, 255, -1)  # 在图像中绘制一个白色圆形

    # 调用 detect_sift_features 函数
    keypoints, descriptors = detect_sift_features(test_image)

    # 检查返回的关键点和描述符是否符合预期
    assert descriptors.shape[1] == 128  # SIFT 描述符的维度应为 128
    assert descriptors.shape[0] == len(keypoints)  # 描述符的数量应与关键点数量相同