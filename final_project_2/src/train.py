import copy
import time
import torch
from pathlib import Path
from torch import nn

from final_project_2.src.data import get_dataloaders
from final_project_2.src.evaluate import evaluate
from final_project_2.src.model import CNN,CIFARResNet,CIFARPlainDeepNet




def count_trainable_parameters(model):
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

def build_model(model_name: str):
    if model_name == "cnn":
        return CNN()
    if model_name == "resnet":
        return CIFARResNet()
    if model_name == "plain":
        return CIFARPlainDeepNet()
    raise ValueError(f"Unknown model name: {model_name}")

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return average_loss, accuracy


def fit(model, train_loader, validation_loader, loss_fn, optimizer, num_epochs=10):
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_fn, optimizer
        )

        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_fn
        )       
        
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss {train_loss:.4f}, train acc {train_accuracy:.2f}% | "
            f"val loss {validation_loss:.4f}, val acc {validation_accuracy:.2f}%"
        )

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    return best_val_loss, best_epoch, best_state , history


def main():
    torch.manual_seed(42)

    train_loader, validation_loader, test_loader = get_dataloaders(
        batch_size=64,
        seed=42,
        train_subset_size=5000,
        validation_subset_size=1_000,
    )
    model_name = "resnet"  # Change to "resnet" to use the ResNet model
    model = build_model(model_name)
    print(f"Using model: {model_name}")
    print(f"Number of trainable parameters: {count_trainable_parameters(model)}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_time = time.time()
    best_val_loss, best_epoch, best_state , history = fit(
        model,
        train_loader,
        validation_loader,
        loss_fn,
        optimizer,
        num_epochs=10,
    )
    model.load_state_dict(best_state)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    
    checkpoint = {
        "model_name": model_name,
        "model_state_dict": best_state,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "seed": 42,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "num_epochs": 10,
        "train_subset_size": 5000,
        "validation_subset_size": 1000,
        "num_parameters": count_trainable_parameters(model),
        "elapsed_time": elapsed_time,
        "history": history,
    }
    checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"best_model_{model_name}.pth"

    torch.save(checkpoint, checkpoint_path)
    print(f"Best model saved to {checkpoint_path}")
    

    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn)

    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
    