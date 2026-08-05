
import cv2


def detect_sift_features(image):
    if image.ndim == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    #建立SIFT特征检测器
    detector = cv2.SIFT_create()
    #检测图像中的关键点和描述符
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)
    return keypoints, descriptors
