# Classical Image Mosaicking

## Project Overview

本專案實作一個不使用 deep learning 的 classical image mosaicking pipeline。

Given two overlapping images，系統會估計兩張影像之間的 geometric relationship，
並將它們 stitched into a single mosaic image。

The main components include feature detection, feature matching,
robust homography estimation, image warping, and image blending。

## Pipeline

```text
Input images A and B
    ↓
SIFT feature detection
    在兩張影像中偵測具有代表性的 keypoints，
    並計算對應的 SIFT descriptors
    ↓
SIFT descriptor matching
    比較 image A 與 image B 的 descriptors，
    找出可能互相對應的 feature pairs
    ↓
Lowe ratio test
    比較每個 descriptor 的 nearest neighbor
    與 second-nearest neighbor 距離，
    移除具有高度 ambiguity 的 matches
    ↓
RANSAC homography estimation
    反覆隨機抽取四組 correspondences，
    估計 candidate homography H，
    並使用全部 correspondences 評估 inlier 數量
    ↓
Reprojection error evaluation
    將 source points 使用 H 投影到 destination image，
    計算 projected points 與 matched destination points
    之間的 pixel distance
    ↓
Canvas construction and image warping
    將 image A 的 corners 使用 H 轉換，
    與 image B 的 corners 一起決定 canvas bounds，
    再將兩張影像 warp 到共同的 canvas coordinate system
    ↓
Mask-based image blending
    使用 valid masks 區分 only-A、only-B 與 overlap regions，
    overlap region 目前使用 equal-weight averaging
    ↓
Mosaic output
    輸出最後的 stitched mosaic image
```


## Project Structure

```text
final_project_3/
├── data/
│   ├── room_left.jpg
│   └── room_right.jpg
├── outputs/
├── scripts/
│   ├── demo_synthetic_mosaic.py
│   ├── demo_feature_match.py
│   └── real_demo.py
├── src/
│   ├── blending.py
│   ├── features.py
│   ├── homography.py
│   ├── matching.py
│   ├── ransac.py
│   └── warping.py
└── tests/
    ├── test_feature.py
    ├── test_homography.py
    └── test_ransac.py
```
## Methods

### 1. Feature Detection and Description

本專案使用 SIFT (Scale-Invariant Feature Transform) 偵測影像中的
local features。

SIFT 的目標是找到對 scale 與 rotation 具有一定 robustness 的
keypoints，並為每個 keypoint 建立一個 128-dimensional descriptor。

在 implementation 中：

1. 將 color image 轉換為 grayscale image
2. 使用 `cv2.SIFT_create()`
3. 呼叫 `detectAndCompute()`
4. 回傳 keypoints 與 descriptors

對於一張影像：

```text
keypoints:   list with K keypoints
descriptors: (K, 128)
```

其中 `K` 是偵測到的 keypoint 數量，每一列 descriptor 對應一個
keypoint。

這個階段只描述影像中的 local appearance，尚未判斷兩張影像中的
keypoints 是否具有正確的 geometric correspondence。

### 2. Feature Matching

對 image A 中的每個 descriptor，使用 Brute-Force Matcher 在 image B
中尋找 nearest neighbor 與 second-nearest neighbor。

本專案使用 Lowe ratio test 篩選 ambiguous matches：

```text
ratio = distance(nearest) / distance(second-nearest)
accept match if ratio < 0.75
```

ratio 越小代表最佳 match 相對更明確。這個步驟只根據 descriptor
similarity 篩選，尚未保證 geometric consistency；後續仍需要 RANSAC。

### 3. Homography Estimation

本專案使用一個 `3 × 3` homography matrix `H` 描述兩張影像之間的
projective transformation：

```text
p_B ~ H p_A
```

其中 `p_A` 與 `p_B` 是 homogeneous coordinates，`~` 表示兩邊只差一個
非零 scale。

Homography matrix 有 8 個 degrees of freedom，因此至少需要四組
non-degenerate point correspondences。每組 correspondence 會產生兩個
linear equations，組成 DLT matrix `A`：

```text
A has shape (2N, 9)
```

使用 Singular Value Decomposition (SVD) 分解 `A`，取最小 singular value
所對應的 right singular vector 作為 homography vector，最後 reshape 成
`3 × 3` matrix。

### 4. RANSAC Homography Estimation

Descriptor matching 仍可能包含 incorrect correspondences，因此不能直接
使用全部 matches 估計 H。

RANSAC 的流程如下：

1. Randomly sample four correspondences
2. 使用 DLT estimate a candidate homography
3. 使用 candidate H 投影全部 source points
4. 計算所有 correspondences 的 reprojection errors
5. 使用 pixel threshold 建立 inlier mask
6. 保留 inlier count 最大的 candidate H
7. 使用最佳 inliers 重新估計 final homography

本專案的 default RANSAC threshold 為 `5.0` pixels。

### 5. Reprojection Error

對一組 correspondence `p_A → p_B`，先使用 H 將 source point 投影到
image B：

```text
p_hat_B = normalize(H p_A)
```

reprojection error 定義為：

```text
e_i = || p_hat_B_i - p_B_i ||_2
```

error 越小代表 homography 對該 correspondence 的幾何解釋越好。
RANSAC 使用這個 error 判斷 inlier 與 outlier。

### 6. Canvas Construction and Image Warping

`H_image2_from_image1` 將 image A 的座標轉換到 image B 的座標系統。

為了建立共同 canvas：

1. 使用 H 轉換 image A 的四個 corners
2. 使用 identity transformation 取得 image B 的 corners
3. 合併兩組 corners 並計算 bounding box
4. 建立 translation matrix，將最小座標平移到 canvas origin
5. 將兩張影像 warp 到同一個 canvas
6. 同時 warp 全白 mask，以取得每張 warped image 的 valid region

這些步驟封裝在 `warp_to_common_canvas()` 中，因此 demo script 不需要
直接處理 canvas translation matrix。

### 7. Mask-based Image Blending

對兩張 warped images 的 valid masks，將像素分成三種區域：

```text
only-A   = mask_A and not mask_B
only-B   = mask_B and not mask_A
overlap  = mask_A and mask_B
```

目前的 baseline blending strategy 為：

- only-A：使用 image A 的像素
- only-B：使用 image B 的像素
- overlap：使用兩張影像的 equal-weight average

這個方法簡單且容易驗證，但當兩張真實影像的 exposure 不同時，
overlap region 可能仍然出現 visible seam。

## Requirements

- Python 3
- NumPy
- OpenCV
- pytest

## How to Run

請從 workspace root 執行下列指令：

```powershell
python -m pytest final_project_3/tests
```

執行 synthetic demos：

```powershell
python -m final_project_3.scripts.demo_synthetic_mosaic
python -m final_project_3.scripts.demo_feature_match
```

執行 real-image mosaicking：

```powershell
python -m final_project_3.scripts.real_demo
```

Real-image demo 預期在 `final_project_3/data/` 中找到：

```text
room_left.jpg
room_right.jpg
```

執行 `real_demo.py` 後，程式會顯示 input images、warped images 與
blended mosaic。按下任意按鍵即可關閉 OpenCV windows。

## Experiments

### Synthetic Image Experiment

Synthetic images 用來驗證 pipeline 的基本幾何與 blending 行為。
第二張影像由已知 translation homography 產生，因此可以比較 estimated H
與 ground-truth H，並檢查 warped images 是否對齊。

### Real Image Experiment

Real-image experiment 使用同一個房間的兩張具有 overlap 的照片。
一次觀察到的結果如下；由於 RANSAC 使用 random sampling，不同執行次數
的數值可能略有變化。

| Metric | Observed value |
|---|---:|
| Candidate matches | 165 |
| Inliers | 72 |
| Inlier ratio | 43.6% |
| Mean inlier error | 1.93 pixels |
| Median inlier error | 1.51 pixels |
| Maximum inlier error | 4.83 pixels |
| Mean outlier error | 908.33 pixels |
| Mosaic canvas shape | `(1637, 2974, 3)` |

The low inlier reprojection error indicates that the selected inliers are
geometrically consistent, while the much larger outlier error shows the value
of RANSAC in rejecting incorrect matches。

## Failure Analysis

### Black Regions and Irregular Boundary

Warping a rectangular image with a projective transformation generally produces
a quadrilateral. The canvas itself is rectangular, but some canvas pixels are
not covered by either warped image, so black triangular regions may appear。

這些 black regions 主要是 canvas boundary effect，不一定代表
homography estimation 失敗。後續可以加入 valid-region cropping 或
largest-rectangle cropping。

### Camera Roll and Perspective Difference

Real photographs may contain camera rotation、camera roll 與不同的 viewpoint。
如果相機在拍攝兩張照片時沒有保持水平，warp 後的影像可能出現傾斜。

此外，書櫃、沙發與牆面位於不同 depth。單一 homography 對 planar scene
或純 camera rotation 最有效；當場景存在明顯 parallax 時，不同深度的物體
可能無法同時完美對齊。

### Matching Outliers

書本封面與書脊可能包含重複或相似紋理，導致 descriptor matching 產生
ambiguous correspondences。低紋理牆面則提供較少可靠的 keypoints。
Lowe ratio test 可以先移除部分 ambiguous matches，RANSAC 再從剩餘
matches 中找出 geometrically consistent inliers。

### Blending Seam

目前 overlap region 使用 equal-weight averaging。當兩張照片有不同的
exposure、亮度或 white balance 時，兩張影像的交界仍可能看得出來。
可使用 feather blending、distance-weighted blending 或 multi-band blending
改善這個問題。

## Limitations and Future Work

- 目前使用 unnormalized DLT；可加入 Hartley normalization 改善數值穩定性
- RANSAC 使用固定 iteration 數與 pixel threshold，尚未加入 adaptive iteration
- 尚未檢查 degenerate four-point samples
- Blending 目前只有 equal-weight averaging
- 尚未自動移除 warped canvas 的 black regions
- 目前只處理兩張影像，尚未建立 multi-image panorama
- 對 large parallax、moving objects 與低 overlap 場景的處理能力有限

Possible extensions include normalized DLT、adaptive RANSAC、feather blending、
multi-band blending 與 multi-image panorama stitching。
