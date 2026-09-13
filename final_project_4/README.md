# Project 4 — Tiny FCOS Object Detection

這個專案以 PyTorch 從核心元件建立一個教育用途的 Tiny FCOS-style object detector，並在 PASCAL VOC 2007 上訓練與評估。目標不只是產生 bounding boxes，而是理解 object detection 的完整資料流、tensor shapes、target assignment、loss、inference、AP/mAP 與 failure analysis。

## 專案範圍

- Dataset：PASCAL VOC 2007，20 classes
- Input：固定 resize 至 `320 × 320`
- Backbone：ImageNet-pretrained ResNet-18
- Multi-scale features：FPN-like `P3/P4/P5`
- Detector：anchor-free、one-stage、FCOS-style head
- Box representation：continuous `xyxy = [xmin, ymin, xmax, ymax]`
- Evaluation：`mAP@0.5`，all-point interpolated AP

本專案是簡化版 FCOS，不是原論文的完整重現。

## 完整 Pipeline

```text
VOC image + XML annotation
        │
        ▼
parse_voc_annotation()
PIL image + raw XML
→ image Tensor[3,H,W], float32, range [0,1]
→ target boxes [N,4], labels [N], difficult [N]
        │
        ▼
resize_image_and_boxes()
→ image [3,320,320]
→ boxes 依 x/y 比例同步縮放
        │
        ▼
detection_collate_fn()
→ images [B,3,320,320]
→ targets list[dict]，保留每張圖片不同的物體數 N_i
        │
        ▼
ImageNet normalization
        │
        ▼
ResNet18Backbone
├── C3 [B,128,40,40]
├── C4 [B,256,20,20]
└── C5 [B,512,10,10]
        │
        ▼
FeaturePyramidNetwork
├── P3 [B,128,40,40], stride 8
├── P4 [B,128,20,20], stride 16
└── P5 [B,128,10,10], stride 32
        │
        ▼
DetectionHead（各層共享 weights）
├── class logits [B,20,H,W]
├── positive ltrb [B,4,H,W]
└── centerness logits [B,1,H,W]
        │
        ▼
flatten_predictions() + generate_locations()
├── class_logits      [B,L,20]
├── box_regression    [B,L,4]
├── centerness_logits [B,L]
├── locations         [L,2]
└── strides           [L]
```

對 `320 × 320` input：

```text
L = 40×40 + 20×20 + 10×10 = 2100 prediction locations
```

## Training Branch

```text
Dataset GT boxes/labels
        │
        ▼
assign_targets_to_locations()
├── location 是否位於 GT box 內
├── GT 尺寸是否符合該 FPN level 的 regression range
├── 多個候選 GT 重疊時選擇面積最小者
└── background location 使用 label = -1
        │
        ▼
assigned targets
├── labels             [L]
├── box_targets        [L,4]
├── centerness_targets [L]
└── positive_mask      [L]
        │
        ▼
compute_detection_loss()
├── Sigmoid Focal Loss：所有 locations 的 classification
├── GIoU Loss：positive locations 的 box regression
└── BCEWithLogits Loss：positive locations 的 centerness
        │
        ▼
backward() → optimizer.step()
```

### 為什麼 classification 必須包含 background

Background location 的 one-hot target 對 20 個 classes 全為零，但仍參與 focal loss。否則模型不會因為在背景上預測物體而受到懲罰。

### 為什麼使用 centerness

同一個 GT box 內可能有許多 positive locations。接近邊界的位置通常較難產生準確框，因此 centerness 會降低這些位置在 inference 時的最終分數：

```text
final score = sigmoid(class_logit) × sigmoid(centerness_logit)
```

### 為什麼將 Smooth L1 改成 GIoU

Smooth L1 直接量測四個 pixel distances 的誤差，沒有直接最佳化 predicted box 與 GT box 的重疊。相同的 pixel error 對小框和大框可能造成完全不同的 IoU。

GIoU loss 先將 `ltrb` 解碼成 `xyxy`：

```text
locations       [L,2]
predicted_ltrb  [B,L,4]
target_ltrb     [B,L,4]
        │
        ▼ decode_boxes()
predicted_xyxy  [B,L,4]
target_xyxy     [B,L,4]
        │ positive_mask [B,L]
        ▼
positive boxes  [P,4]
        │
        ▼
GIoU loss       scalar []
```

GIoU 更直接對齊最終以 IoU 判定 TP/FP 的 evaluation objective。

## Inference Branch

```text
raw model predictions
        │
        ├── sigmoid(class_logits)
        ├── sigmoid(centerness_logits)
        └── decode ltrb → xyxy boxes
        │
        ▼
class probability × centerness probability
        │
        ▼
score threshold → top-k candidates
        │
        ▼
class-aware batched NMS
        │
        ▼
boxes [K,4], labels [K], scores [K]
```

Class-aware NMS 只抑制相同 class 的重疊框。若同一 location 對多個 classes 都有高分，相同座標仍可能以不同 labels 被保留；這屬於 classification ambiguity，不代表 NMS 實作錯誤。

## Evaluation

每個 class 分開處理：

1. 依 prediction score 由高至低排序。
2. 以 IoU 將 prediction greedy match 至尚未匹配的 GT。
3. 第一個符合 threshold 的 prediction 為 TP；重複匹配同一 GT 為 FP。
4. 與 `difficult=True` GT 匹配的 prediction 會被忽略。
5. 累積 TP/FP 得到 precision–recall curve。
6. 使用 all-point interpolation 計算 AP，再對有 GT 的 classes 取平均得到 mAP。

目前主要比較使用 validation set 的前 `100` batches（batch size 2，最多 200 images），以維持各次實驗的評估條件一致。因此這些數字不是完整 VOC 2007 validation-set result。

## 實驗設定

```text
Input size:          320 × 320
Batch size:          2
Optimizer:           AdamW
Learning rate:       1e-4
Backbone:            ImageNet-pretrained ResNet-18
Input normalization: ImageNet mean/std
Regression ranges:   stride 8  → [0,64]
                     stride 16 → [64,128]
                     stride 32 → [128,∞]
```

## 實驗結果

| Experiment | Box loss | Training budget | mAP@0.5 |
|---|---:|---:|---:|
| `pretrained_smooth_l1` | Smooth L1 | 3 epochs × 500 steps | 0.0284 |
| `pretrained_giou` | GIoU | 3 epochs × 500 steps | 0.0864 |
| `pretrained_giou_longer` | GIoU | 5 full epochs | **0.2678** |

在相同短訓練條件下，GIoU 將 mAP@0.5 從 `0.0284` 提升至 `0.0864`：

```text
absolute improvement = 0.0580
relative improvement ≈ 3.04×
```

這支持「overlap-based regression objective 比 raw pixel-distance Smooth L1 更適合目前 detector」的假設，但不表示模型已達到良好的絕對 detection performance。

將 GIoU training budget 從 `1500` optimizer steps 增加為約 `6255` steps 後：

```text
mAP@0.5: 0.0864 → 0.2678
absolute improvement = 0.1814
relative improvement ≈ 3.10×
```

這說明 short GIoU model 明顯 undertrained。相較 Smooth L1 short run，最終 mAP 約為 `9.43×`；但這個差異同時包含 box loss 與 training budget 兩個變因，因此不能全部歸因於 GIoU。

較早的 scratch experiment 曾得到約 `0.0058` mAP，但它與後續實驗同時存在多項設定差異，因此不視為嚴格的單變因 ablation。

## Failure Analysis

### 1. Localization 改善，但部分框仍未跨過 IoU=0.5

在固定 sample 上：

```text
Smooth L1 prediction 的最高 IoU：約 0.146
GIoU short prediction 的最高 IoU：約 0.396
GIoU longer prediction 的最高 IoU：約 0.675
```

Longer model 的固定 sample 共有 4 個 predictions，其中只有一個框的 IoU 達到 `0.675`。其餘三個 prediction 的最高 IoU 為 `0.235`、`0.165`、`0.076`，無論 class 是否正確都無法成為 TP。

從 GT 角度觀察，每個 GT 能取得的最佳 IoU 約為：

```text
[0.235, 0.675, 0.000, 0.165, 0.028]
```

因此 aggregate mAP 已大幅改善，但這張 sample 仍呈現明顯的 localization 與 recall failure。

### 2. Classification ambiguity

Short model 曾將完全相同的 boxes 同時預測為 `chair`、`person`、`tvmonitor` 等不同 classes。原因是每個 location 具有 20 個獨立 sigmoid class scores，而 class-aware NMS 不會跨 class 抑制。

Longer model 在固定 sample 的 4 個 predictions 全部輸出 `chair`，表示該 sample 的跨類別 ambiguity 已明顯降低；但這不能單獨證明所有圖片都具有相同改善。

提高 score threshold 或改用 class-agnostic NMS 可以讓 visualization 看起來較乾淨，但無法修正分類能力，也可能掩蓋真正的模型問題。

### 3. Small-object failure

固定 sample 中最小 GT 的最佳 prediction IoU 約為 `0.028`，另一個 GT 完全沒有重疊 prediction。Tiny FPN 只使用 `P3/P4/P5`、輸入固定為 `320 × 320`，小物體可用的特徵解析度與訓練訊號有限。

### 4. Per-class performance 不平均

GIoU longer run 的代表性 AP：

```text
car:        0.7091
aeroplane:  0.5269
horse:      0.5038
dog:        0.4686
tvmonitor:  0.4026
person:     0.3951
```

19 個 classes 已得到非零 AP，但 `bottle` 仍為零。細長、小型物體仍是目前模型的主要弱點；少量 GT 也會使個別 class AP 對少數 prediction 特別敏感。

### 固定 sample visualization

綠框為 GT，紅框為 `pretrained_giou_longer` predictions：

![Tiny FCOS GIoU longer prediction](assets/pretrained_giou_longer_prediction.jpg)

### 5. 實驗限制

- 固定 `320 × 320` resize 會改變原始 aspect ratio。
- 沒有 data augmentation、learning-rate scheduler 或 multi-scale training。
- 只使用 `P3/P4/P5`，沒有完整 FCOS 的額外 pyramid levels 與其他訓練技巧。
- Training target assignment 仍包含 `difficult` objects，但 evaluation 會忽略 difficult GT；這是一項明確的 policy difference。
- 尚未固定 random seed，因此不同次重新訓練仍包含 initialization 與 data-order variation。
- 目前比較只使用 validation subset，不代表正式 benchmark result。

## 專案結構

```text
final_project_4/
├── assets/              # README 使用的可視化結果
├── src/
│   ├── boxes.py          # box area、IoU、clip、decode
│   ├── transforms.py     # image/box resize
│   ├── data.py           # VOC parser、Dataset、collate
│   ├── model.py          # backbone、FPN、detection head
│   ├── detector.py       # end-to-end FCOSDetector
│   ├── targets.py        # locations、matching、assignment
│   ├── losses.py         # focal、GIoU、centerness losses
│   ├── inference.py      # scoring、threshold、NMS
│   ├── metrics.py        # TP/FP matching、AP、mAP
│   ├── evaluation.py     # dataset-level detection evaluation
│   ├── training.py       # train/evaluate loops
│   └── checkpoint.py     # versioned checkpoint schema
├── scripts/
│   ├── inspect_data.py
│   ├── smoke_train.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inspect_pred.py
│   └── overfit_one_image.py
└── tests/
```

## 執行方式

從 workspace root 執行：

```powershell
.\.venv\Scripts\python.exe -m pytest final_project_4/tests -q
```

```powershell
.\.venv\Scripts\python.exe -m final_project_4.scripts.smoke_train
```

```powershell
.\.venv\Scripts\python.exe -m final_project_4.scripts.train
```

```powershell
.\.venv\Scripts\python.exe -m final_project_4.scripts.evaluate
```

```powershell
.\.venv\Scripts\python.exe -m final_project_4.scripts.inspect_pred
```

Training checkpoints 與 local visualizations 儲存在 `outputs/`，不納入版本控制。

## 下一步

- 完成涵蓋 motivation、shape、assignment、loss、inference、evaluation 與 failure analysis 的 100 分理解測驗。
