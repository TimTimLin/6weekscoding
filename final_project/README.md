# Fashion-MNIST CNN Classification

使用 PyTorch 建立 CNN，將 Fashion-MNIST 灰階服飾圖片分類成 10 個類別。

## Project Goal

本專案練習完整的影像分類流程：

- Dataset 與 DataLoader 建立
- Train / Validation / Test 分離
- CNN 模型訓練
- 依 validation loss 選擇最佳模型
- Confusion matrix 與 per-class accuracy 分析
- 使用單元測試驗證資料、模型與評估函式

## Dataset

使用 Fashion-MNIST：

- 灰階圖片，大小為 `28 × 28`
- 10 個服飾類別
- Train：54,000 張
- Validation：6,000 張
- Official test：10,000 張
- Transform：`ToTensor()`，像素範圍為 `[0, 1]`

測試集只在模型選擇完成後使用，不參與模型選擇，避免資料洩漏。

## Project Structure

```text
final_project/
├── src/
│   ├── data.py          # Dataset 與 DataLoader
│   ├── model.py         # CNN 模型
│   ├── train.py         # 訓練流程
│   └── evaluate.py      # 評估與錯誤分析
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_evaluate.py
├── data/                # 本機資料，不提交到 Git
└── README.md
```

## Environment and Usage

本專案使用 Python 3.12 與 CPU 版 PyTorch。

從 workspace 根目錄執行：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m final_project.src.train
```

Fashion-MNIST 會在第一次執行時自動下載到 `final_project/data/`。

執行測試：

```powershell
python -m pytest final_project/tests -q
```

## Method

模型由兩個 CNN block 組成：

```text
Conv2d → ReLU → MaxPool2d
Conv2d → ReLU → MaxPool2d
Flatten → Linear
```

訓練設定：

- Loss：`CrossEntropyLoss`
- Optimizer：`Adam`
- Learning rate：`0.001`
- Batch size：`64`
- Epochs：`5`
- 最佳模型：使用最低 validation loss 的 epoch

## Experiments

比較 baseline 與增加 channel 數量的 CNN，其他訓練條件維持相同。

| Model | Channels | Best Epoch | Best Validation Loss | Test Accuracy | Shirt Accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | `8 → 16` | 5 | 0.3321 | 88.02% | 58.90% |
| Wider CNN | `16 → 32` | 4 | 0.2905 | 89.03% | 68.50% |

增加模型容量後，整體 test accuracy 提升約 1.01 個百分點，Shirt 類別提升約 9.60 個百分點。

## Error Analysis

Wider CNN 的主要 Shirt 錯誤如下：

- Shirt → T-shirt/top：135 張
- Shirt → Coat：82 張
- Shirt → Pullover：68 張

這些類別都屬於外觀相近的上半身服飾，因此容易產生混淆。Shirt 雖然因為增加模型容量而改善，仍然是較困難的類別。

## Tests

目前測試涵蓋：

- Model output shape
- Dataset 長度、batch shape、dtype 與像素範圍
- Confusion matrix 的重複索引累加

後續會繼續補充 `model`、`data`、`evaluate` 與 `train` 的單元測試。

## Limitations and Future Work

- 模型只訓練 5 個 epochs
- 只比較了一種模型容量變化
- 尚未使用資料增強、BatchNorm 或 transfer learning
- Shirt、T-shirt/top、Pullover 與 Coat 仍然容易混淆
- Dataset 不包含在 Git repository 中，需在本機重新下載
