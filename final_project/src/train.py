from final_project.src.model import FashionCNN
from final_project.src.data import get_dataloaders
from final_project.src.evaluate import  (
    evaluate, 
    collect_predictions,
    build_confusion_matrix, 
    print_confusion_report,
)
import torch
from torch import nn
import copy

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()  
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_images, batch_labels in train_loader:
        optimizer.zero_grad()
        logits = model(batch_images)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*batch_images.size(0)  # Accumulate loss
        correct += (logits.argmax(dim=1) == batch_labels).sum().item()
        total += batch_labels.size(0)
    average_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return average_loss, accuracy

def fit(model, train_loader, validation_loader, loss_fn, optimizer, num_epochs=5):
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    for i in range(num_epochs):
        average_loss, accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, loss_fn)
        print(f"Epoch {i+1}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%")
        # 更新最佳模型權重
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_epoch = i + 1
            best_state = copy.deepcopy(model.state_dict())
    print(f"Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    return best_val_loss, best_epoch, best_state

def main() -> None:
    torch.manual_seed(42)
    train_loader, validation_loader, test_loader = get_dataloaders(batch_size=64, seed=42)
    model = FashionCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss, best_epoch, best_state = fit(model, train_loader, validation_loader, loss_fn, optimizer, num_epochs=5)
    
    model.load_state_dict(best_state)
    
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    predictions, true_labels = collect_predictions(model,test_loader)
    
    confusion = build_confusion_matrix(true_labels,predictions, num_classes=10)
    
    print_confusion_report(confusion,target_class_index=6, k=3)
    
    
if __name__ == "__main__":
    main()