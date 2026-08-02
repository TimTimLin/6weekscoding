from pathlib import Path
import torch
from torch.utils.data import DataLoader
from final_project_2.src.data import get_dataloaders
from final_project_2.src.evaluate import CLASS_NAMES,build_confusion_matrix, collect_predictions
from final_project_2.src.model import CIFARResNet

model=CIFARResNet()
# 載入最佳模型的checkpoint
checkpoint_path = Path(__file__).resolve().parents[1] / "checkpoints" / "best_model_resnet.pth"
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
# 載入模型權重
model.load_state_dict(checkpoint["model_state_dict"])
# 一樣load optimizer狀態
train_loader, validation_loader, test_loader = get_dataloaders(batch_size=64, seed=42)
predictions, true_labels = collect_predictions(model, test_loader)
confusion_matrix = build_confusion_matrix(true_labels, predictions, num_classes=10)
per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(dim=1)
print(confusion_matrix)
print(per_class_accuracy)
for c, class_name in enumerate(CLASS_NAMES):
    class_index = c
    row = confusion_matrix[c].clone()
    row[class_index] = 0
    top_values, top_indices = row.topk(3)
    top_class_names = [CLASS_NAMES[i] for i in top_indices.tolist()]
    
    print(f"Class: {class_name}, Accuracy: {per_class_accuracy[c]:.4f}")
    for i in range(3):
        print(f"    {class_name} -> {top_class_names[i]} : {top_values[i].item()}")
    
    # Example output:
    # airplane -> ship: 324
    # airplane -> truck: 250
    # airplane -> deer: 71