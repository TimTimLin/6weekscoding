import cv2
import numpy as np
from final_project_3.src.features import detect_sift_features
from pathlib import Path
from final_project_3.src.matching import match_features, extract_matched_points
from final_project_3.src.ransac import ransac_homography
from final_project_3.src.warping import warp_to_common_canvas
from final_project_3.src.blending import blend_images
from final_project_3.src.homography import compute_reprojection_errors,estimate_homography_dlt_normalized 

def load_image(path):
    encoded = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def save_image(path, image):
    path = Path(path)

    success, encoded = cv2.imencode(
        path.suffix,
        image,
    )

    if not success:
        return False

    path.write_bytes(encoded.tobytes())
    return True

def show_image(title, image, width=800, height=600):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, width, height)
    cv2.imshow(title, image)

data_dir = Path(__file__).resolve().parents[1] / "data"
output_dir = Path(__file__).resolve().parents[1] / "outputs"

image1 = load_image(data_dir / "room_left.jpg")
image2 = load_image(data_dir / "room_right.jpg")
if image1 is None or image2 is None:
    raise FileNotFoundError("Could not load images. Please check the paths.") 

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

if H_estimated is None:
    raise ValueError("RANSAC failed to find a valid homography. Please check the input points and matches.")

H_normalized_estimated = estimate_homography_dlt_normalized(
    points1[inlier_mask],
    points2[inlier_mask]
)

H_estimated_errors = compute_reprojection_errors(
    points1,
    points2,
    H_estimated,
)

H_normalized_estimated_errors = compute_reprojection_errors(
    points1,
    points2,
    H_normalized_estimated,
)



print("mean inlier error:", H_estimated_errors[inlier_mask].mean())
print("median inlier error:", np.median(H_estimated_errors[inlier_mask]))
print("mean normalized inlier error:", H_normalized_estimated_errors[inlier_mask].mean())
print("median normalized inlier error:", np.median(H_normalized_estimated_errors[inlier_mask]))


(warped_image_a, valid_mask_a), (warped_image_b, valid_mask_b) = warp_to_common_canvas(
    image1,
    image2,
    H_normalized_estimated,
)

blended_image = blend_images(warped_image_a, warped_image_b, valid_mask_a, valid_mask_b)
print("matches:", len(matches))
print("inliers:", int(inlier_mask.sum()), "/", len(inlier_mask))
print("inlier ratio:", inlier_mask.mean())
print("canvas shape:", blended_image.shape)


show_image("Blended Image", blended_image)
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "real_mosaic_normalized.jpg"

saved = save_image(output_path, blended_image)

print("saved:", saved)
print("output path:", output_path.resolve())

cv2.waitKey(0)
cv2.destroyAllWindows()
