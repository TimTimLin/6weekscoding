from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from final_project_5.src.data import build_oxford_pet_dataset
from final_project_5.src.transforms import SegmentationTransform
from matplotlib import pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parents[1]

transform = SegmentationTransform(size=(256, 256), horizontal_flip_prob=0.0)

dataset = build_oxford_pet_dataset(
    root=PROJECT_ROOT / "data",
    split="trainval",
    joint_transform=transform,
    download=False,
)


# Real-data visual check─ RGB image 正常嗎？
            # ├─ mask 區域正常嗎？
            # └─ overlay 真的貼在 pet 上嗎？ ← 最重要
image, mask = dataset[0]

image_array = image.permute(1, 2, 0).numpy()  # 將 [C,H,W] 轉換為 [H,W,C] 符合 matplotlib 的格式
mask_array = mask.numpy()  # 將 [H,W] 轉換為 numpy
class_colors = np.array(
    [
        [255, 0, 0],      # 0: foreground
        [0, 0, 255],      # 1: background
        [255, 255, 0],    # 2: not_classified
    ],
    dtype=np.uint8,
)
# 將 mask_array 的類別索引映射到對應的 RGB 顏色
# 0 → class_colors[0] → [255, 0, 0]
# 1 → class_colors[1] → [0, 0, 255]
# 2 → class_colors[2] → [255, 255, 0]
# ex ample: mask_array = [[0, 1], [2, 0]] → mask_rgb = [[[255, 0, 0], [0, 0, 255]], [[255, 255, 0], [255, 0, 0]]]
mask_rgb = class_colors[mask_array]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# 畫出三個子圖：原始 RGB 圖像、目標遮罩、以及圖像與遮罩的疊加
axes[0].imshow(image_array)
axes[0].set_title("RGB Image")

axes[1].imshow(mask_rgb)
axes[1].set_title("Target Mask")

axes[2].imshow(image_array)
axes[2].imshow(mask_rgb, alpha=0.35)
axes[2].set_title("Image + Mask Overlay")

for ax in axes:
    ax.axis("off")

fig.tight_layout()
output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(
    output_dir / "transformed_sample.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)


# 建立 DataLoader 以檢查批次資料的形狀和類型

dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
images, masks = next(iter(dataloader))

print(type(images))
print(images.shape)
print(images.dtype)
print(images.min().item(), images.max().item())

print(type(masks))
print(masks.shape)
print(masks.dtype)
print(torch.unique(masks))
