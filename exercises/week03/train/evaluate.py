import torch
from torch import nn
def evaluate(model, dataloader, loss_fn) -> tuple[float, float]:
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation
        for batch in dataloader:
            images, labels = batch
            logits = model(images)  # Forward pass
            loss = loss_fn(logits, labels)  # Compute loss
            predicted = torch.argmax(logits, dim=1)  # Get predicted class
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)  # Accumulate total loss 
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy