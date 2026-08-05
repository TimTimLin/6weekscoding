import cv2
import numpy as np
from final_project_3.src.blending import blend_images
from final_project_3.src.warping import transform_corners, compute_canvas_bounds, finalize_canvas_bounds, warp_image_to_canvas
image_a=np.zeros((100,160,3),dtype=np.uint8)
image_b=np.zeros((100,160,3),dtype=np.uint8)
image_a=cv2.rectangle(
    image_a,
    (20, 20),
    (80, 80),
    (255, 0, 0),
    -1,
)
image_b=cv2.circle(
    image_b,
    (80, 50),
    25,
    (0, 255, 0),
    -1,
)
H_A = np.array([
    [1, 0, 20],
    [0, 1, 20],
    [0, 0, 1],
], dtype=np.float64)
H_B = np.array([
    [1, 0, 100],
    [0, 1, 20], 
    [0, 0, 1],
], dtype=np.float64)

canvas_size = (260, 140)  # width, height
warped_a, mask_a = warp_image_to_canvas(image_a, H_A, canvas_size)
warped_b, mask_b = warp_image_to_canvas(image_b, H_B, canvas_size)
blended = blend_images(warped_a, warped_b, mask_a, mask_b)
print(warped_a.shape)
print(warped_b.shape)
print(mask_a.shape)
print(mask_b.shape)

print("A valid:", mask_a.sum())
print("B valid:", mask_b.sum())
print("overlap:", (mask_a & mask_b).sum())
print("union:", (mask_a | mask_b).sum())
print(blended[60, 60])    # 藍色矩形，應接近 [255, 0, 0]
print(blended[70, 160])   # overlap 中的綠色圓，應是暗綠色
print(blended[70, 180])   # 只有 B，有效綠色，應接近 [0, 255, 0]