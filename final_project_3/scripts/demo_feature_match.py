import cv2
import numpy as np
from final_project_3.src.features import detect_sift_features
from final_project_3.src.matching import match_features, extract_matched_points
from final_project_3.src.ransac import ransac_homography
from final_project_3.src.warping import warp_to_common_canvas
from final_project_3.src.blending import blend_images
image1= np.zeros((300, 400, 3), dtype=np.uint8)
# 在 base 图像上绘制一些几何形状
cv2.rectangle(image1, (50, 50), (150, 150), (255, 255, 255), -1)
cv2.circle(image1, (280, 100), 40, (255, 255, 255), -1)
cv2.line(image1, (40, 250), (350, 220), (255, 255, 255), 3)

cv2.putText(
    image1,
    "SIFT",
    (160, 260),
    cv2.FONT_HERSHEY_SIMPLEX,
    2,
    (255, 255, 255),
    3,
)
# 做第二張圖像，對 base 圖像進行平移
H = np.array([
    [1, 0, 30],
    [0, 1, 20],
    [0, 0, 1],
], dtype=np.float64)

image2 = cv2.warpPerspective(
    image1,
    H,
    (400, 300),
)
keypoints1, descriptors1 = detect_sift_features(image1)

keypoints2, descriptors2 = detect_sift_features(image2)

matches = match_features(
    descriptors1,
    descriptors2,
)

points1, points2 = extract_matched_points(
    keypoints1,
    keypoints2,
    matches,
)
H_estimated,inlier_mask = ransac_homography(
    points1,
    points2,
)
(warped_image_a, valid_mask_a), (warped_image_b, valid_mask_b) = warp_to_common_canvas(
    image1,
    image2,
    H_estimated
)

blended_image = blend_images(warped_image_a, warped_image_b, valid_mask_a, valid_mask_b)

cv2.imshow("Image 1", image1)
cv2.imshow("Image 2", image2)
cv2.imshow("Warped Image 1", warped_image_a)
cv2.imshow("Warped Image 2", warped_image_b)
cv2.imshow("Blended Image", blended_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
