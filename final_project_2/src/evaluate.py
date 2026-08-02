import torch


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images)
            loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return average_loss, accuracy


def collect_predictions(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            predictions = model(images).argmax(dim=1)
            all_predictions.append(predictions)
            all_labels.append(labels)

    return torch.cat(all_predictions), torch.cat(all_labels)


def build_confusion_matrix(true_labels, predictions, num_classes=10):
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for true_label, prediction in zip(true_labels, predictions):
        confusion_matrix[true_label, prediction] += 1

    return confusion_matrix
