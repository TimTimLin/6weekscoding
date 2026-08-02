# CIFAR-10 CNN vs ResNet-like

比較一般 CNN、Plain deep network 與 ResNet-like model 在 CIFAR-10 上的分類效果，並透過 residual connection ablation 與 confusion matrix 進行分析。

## Project objectives

- 建立可重現的 CIFAR-10 image classification pipeline。
- 比較一般 CNN 與 ResNet-like model 的分類表現。
- 實作 pre-activation residual block 與 projection shortcut。
- 使用 plain deep model 做 residual connection 的初步 ablation。
- 使用 validation loss 選擇最佳 checkpoint，並進行 test evaluation 與錯誤分析。

## Project structure

~~~text
final_project_2/
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
├── scripts/
│   ├── inspect_data.py
│   ├── analyze_checkpoint.py
│   └── plot_history.py
├── tests/
├── checkpoints/
├── outputs/
└── README.md
~~~

## Dataset

使用 CIFAR-10：

- 10 個 classes
- RGB images
- image size: "32 × 32"
- official training set: 50,000 images
- official test set: 10,000 images

完整實驗將 training set 分成：

~~~text
train:      45,000
validation:  5,000
test:       10,000
~~~

輸入影像使用 CIFAR-10 的 mean/std 進行 normalization：

~~~text
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
~~~

目前沒有使用 data augmentation。

## Models

### CNN baseline

~~~text
Conv 3 → 16
ReLU
MaxPool
Conv 16 → 32
ReLU
MaxPool
Flatten
Linear 32 × 8 × 8 → 10
~~~

Trainable parameters: 25,578

### CIFARResNet

~~~text
Stem: Conv 3 → 32

Stage 1: 32 channels, 2 residual blocks
Stage 2: 64 channels, 2 residual blocks, first block uses stride=2
Stage 3: 128 channels, 2 residual blocks, first block uses stride=2

Adaptive Average Pooling
Linear 128 → 10
~~~

Shape transition：

~~~text
[B, 3, 32, 32]
→ [B, 32, 32, 32]
→ [B, 64, 16, 16]
→ [B, 128, 8, 8]
→ [B, 128]
→ [B, 10]
~~~

The residual branch follows a pre-activation style：

~~~text
BatchNorm
→ ReLU
→ Conv 3 × 3
→ BatchNorm
→ ReLU
→ Conv 3 × 3
~~~

For shape-preserving blocks, the shortcut is identity mapping. When the channel count or spatial resolution changes, a 1 × 1 Conv projection shortcut is used.

Trainable parameters: 697,098

### PlainDeepNet

PlainDeepNet uses the same stem, stages, channels and convolutional paths as CIFARResNet, but removes the residual addition. It is used as an initial ablation model.

Trainable parameters: 686,666

## Training setup

Default training configuration used for the full baseline：

~~~text
optimizer: Adam
learning rate: 1e-3
batch size: 64
epochs: 10
seed: 42
loss: CrossEntropyLoss
~~~

The model checkpoint is selected using the lowest validation loss. The test set is evaluated only after loading the best validation checkpoint.

Each checkpoint also stores the per-epoch training and validation history so that learning curves can be reproduced after training.

## Full-data results

### CNN vs ResNet-like

| Model | Parameters | Best val loss | Best epoch | Test loss | Test accuracy |
|---|---:|---:|---:|---:|---:|
| CNN | 25,578 | 0.9146 | 8 | 0.9303 | 68.62% |
| CIFARResNet | 697,098 | 0.5371 | 7 | 0.5659 | 81.46% |

The ResNet-like model improved test accuracy by approximately 12.84 percentage points compared with the CNN baseline.

這些 full-data 數值來自較早的 run；由於當時尚未修正 test evaluation 必須載入最佳 checkpoint 的流程，因此重新執行 full-data experiment 後，才能將它們作為正式 final result。這裡保留它們作為 historical reference。

這個比較也不能完全 isolate residual connections，因為模型在 depth、channel 數、BatchNorm 使用方式與 parameter count 上都有差異。

## 10-epoch CPU ablation

為了在 CPU 上進行較長訓練，這組實驗使用：

```text
train:       5,000
validation:  1,000
test:       10,000
epochs:         10
batch size:     64
seed:            42
```

| Model | Parameters | Training time | Best val loss | Best epoch | Test loss | Test accuracy |
|---|---:|---:|---:|---:|---:|---:|
| PlainDeepNet | 686,666 | 188.87 s | 1.2913 | 10 | 1.3587 | 52.78% |
| CIFARResNet | 697,098 | 162.43 s | 1.1838 | 10 | 1.1859 | 59.84% |

在這組設定下，CIFARResNet 比 PlainDeepNet 高：

```text
59.84% - 52.78% = 7.06 percentage points
```

ResNet 同時具有較低的 training loss、validation loss 與 test loss。不過這個結果不能完全歸因於 residual addition，因為 ResNet 還包含 projection shortcut，並且多了 10,432 個 trainable parameters。

兩個模型在第 10 epoch 才得到最佳 validation loss，表示這組實驗尚未顯示明確 plateau；若增加 epochs，結果可能繼續改變。

## Learning curves

### ResNet-like model

ResNet 的 training loss 持續下降，validation loss 整體下降但有波動；validation accuracy 在第 10 epoch 達到 60.40%。

![ResNet loss curve](outputs/loss_curve_resnet.png)

![ResNet accuracy curve](outputs/accuracy_curve_resnet.png)

### PlainDeepNet

PlainDeepNet 的 training loss 持續下降，但 validation loss 在第 9 epoch 上升，之後於第 10 epoch 回落；validation accuracy 第 10 epoch 為 54.10%。

![PlainDeepNet loss curve](outputs/loss_curve_plain.png)

![PlainDeepNet accuracy curve](outputs/accuracy_curve_plain.png)

## Quick ablation

For CPU-friendly iteration, a smaller experiment was performed with：

~~~text
train:      5,000
validation: 1,000
test:       10,000
epochs:     3
batch size: 64
seed:       42
~~~

| Model | Parameters | Training time | Best val loss | Best epoch | Test accuracy |
|---|---:|---:|---:|---:|---:|
| CNN | 25,578 | 8.19 s | 1.3991 | 3 | 47.92% |
| PlainDeepNet | 686,666 | 111.86 s | 1.5269 | 3 | 40.82% |
| CIFARResNet | 697,098 | 106.25 s | 1.5959 | 2 | 39.05% |

The quick ablation is used for development and debugging only. Its accuracy should not be compared directly with the full-data results.

The small experiment also shows that residual connections do not guarantee higher accuracy in every short training run. Their expected benefit is primarily easier optimization of deeper networks, and a reliable ablation would require longer training and multiple random seeds.

## Error analysis

The following confusion matrix was generated from the quick ResNet checkpoint. Rows represent true labels and columns represent predicted labels.

Per-class accuracy：

| Class | Accuracy |
|---|---:|
| airplane | 16.20% |
| automobile | 34.20% |
| bird | 0.50% |
| cat | 19.10% |
| deer | 52.10% |
| dog | 52.80% |
| frog | 52.90% |
| horse | 31.20% |
| ship | 52.80% |
| truck | 78.70% |

Most frequent confusion pairs：

| True class | Predicted class | Count |
|---|---|---:|
| automobile | truck | 585 |
| bird | deer | 423 |
| cat | dog | 377 |
| airplane | ship | 324 |
| horse | dog | 292 |
| airplane | truck | 250 |
| ship | truck | 250 |
| frog | deer | 197 |

The quick model strongly favors the truck class and has difficulty distinguishing visually similar animal classes. The very low bird accuracy also indicates that the model is under-trained for this quick setting; this result should not be interpreted as the final capability of the full-data ResNet model.

## Checkpoints

Best checkpoints are saved under：

~~~text
final_project_2/checkpoints/
~~~

Each checkpoint stores the model name, model state, best validation loss, best epoch, experiment configuration metadata and per-epoch training history.

## How to run

From the workspace root, activate the shared virtual environment：

~~~powershell
./.venv/Scripts/Activate.ps1
~~~

Run the training module：

~~~powershell
python -m final_project_2.src.train
~~~

The current default configuration is the CPU-friendly quick experiment. To switch models, change model_name in src/train.py：

~~~python
model_name = "cnn"
model_name = "resnet"
model_name = "plain"
~~~

To reproduce the full-data baseline, remove the subset limits and set num_epochs=10.

Run the checkpoint error analysis script after a checkpoint has been created：

~~~powershell
python -m final_project_2.scripts.analyze_checkpoint
~~~

`plot_history.py` 會讀取 checkpoint 中的 history 並產生 loss/accuracy curves。若要繪製另一個模型，需先將腳本中的 checkpoint path 指向對應檔案，再執行：

~~~powershell
python -m final_project_2.scripts.plot_history
~~~

## Validation status

- `data.py` import 不會自動建立 DataLoader 或產生 import side effect。
- model output shape tests 已通過。
- 最新 pytest 結果：`3 passed`。
- confusion matrix 對角線總和與 test accuracy 一致。
- training 與 validation history 已保存到 checkpoint。

## Limitations and next steps

- No data augmentation is currently used.
- The full-data ablation for PlainDeepNet was not run because CPU training is expensive.
- The quick ablation uses only one random seed.
- The 10-epoch ablation uses 5,000 training images and is intended for CPU-friendly comparison.
- PlainDeepNet and CIFARResNet do not have exactly matched parameter counts.
- Training currently uses CPU; GPU execution can be added through a device abstraction.
- More rigorous conclusions require multiple seeds, learning-rate scheduling, parameter-matched ablations and controlled full-data experiments.
- The earlier full-data reference results should be rerun with the corrected best-checkpoint evaluation flow.
- A future project will connect this pipeline to digital image processing tasks such as denoising, restoration, image registration or image compression.

## Conclusion

本專案完成了一個可重現的 CIFAR-10 classification baseline，並實作了 pre-activation residual block、identity shortcut 與 projection shortcut。

在目前的 10-epoch CPU ablation 中，ResNet-like model 的 test accuracy 為 `59.84%`，高於 PlainDeepNet 的 `52.78%`。這支持以下初步觀察：

> 在相同資料與訓練設定下，加入 residual connection 的較深模型具有較好的最佳化與分類表現。

但由於模型參數量、projection shortcut、單一 seed 與較小的 training subset 等因素，這仍是 initial ablation，不能視為對 residual connection 的完全因果隔離。
