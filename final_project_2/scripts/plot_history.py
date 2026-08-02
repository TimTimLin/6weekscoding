from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from final_project_2.src.data import get_dataloaders
from final_project_2.src.evaluate import CLASS_NAMES,build_confusion_matrix, collect_predictions
from final_project_2.src.model import CIFARResNet

model=CIFARResNet()
# 載入最佳模型的checkpoint
checkpoint_path = Path(__file__).resolve().parents[1] / "checkpoints" / "best_model_plain.pth"
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
history = checkpoint["history"]
epochs = range(1, checkpoint["num_epochs"] + 1)
# 繪製訓練和驗證的損失曲線
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["validation_loss"], label="Validation Loss")

plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
output_dir = Path(__file__).resolve().parents[1] /"outputs"
plt.savefig(output_dir / f"loss_curve_{checkpoint['model_name']}.png")
plt.clf()

###############################################
plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
plt.plot(epochs, history["validation_accuracy"], label="Validation Accuracy")  

plt.title("Accuracy Curve")
plt.xlabel("Epoch") 
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.savefig(output_dir / f"accuracy_curve_{checkpoint['model_name']}.png")
plt.clf()   