
import torch
CLASS_NAMES = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot",]

def evaluate(model, dataloader, loss_fn):
    model.eval()  
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  
        for batch in dataloader:
            images, labels = batch
            logits = model(images)  
            loss = loss_fn(logits, labels)  
            predicted = torch.argmax(logits, dim=1)  
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)  
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy

def collect_predictions(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions)
            all_labels.append(labels)
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels) 
    return predictions, labels
def build_confusion_matrix(true_labels,predictions, num_classes=10):
    #建立大小為(num_classes, num_classes)的矩陣，初始值為0
    # 
    #           true_label類別1 true_label類別2 ... true_label類別10
    #   預測類別1          
    #   預測類別2
    #   ...
    #   預測類別10
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    #累加每個預測類別和真實類別的組合出現的次數
    for true_label, prediction in zip(true_labels, predictions):
        confusion_matrix[true_label, prediction] += 1
    return confusion_matrix
def classwise_accuracy(confusion_matrix):
    #計算每個類別的準確率
    #準確率 = 正確預測的數量 / 該類別的總數量
    correct_predictions = torch.diag(confusion_matrix)
    total_per_class = confusion_matrix.sum(dim=1)
    accuracy_per_class = correct_predictions.float() / total_per_class.float()
    return accuracy_per_class
def get_top_confused_classes(confusion_matrix, class_index: int, k: int = 3):
    #計算每個類別的被預測錯誤的次數，並找出次數最多的前top_n個類別
    row = confusion_matrix[class_index].clone()
    # 將該類別的正確預測次數設為0，避免被選中 example: class_index=0, row=[5, 2, 3, 0, 1], after row[class_index]=0 -> row=[0, 2, 3, 0, 1]
    row[class_index] = 0
    # 找出次數最多的前k個類別 example: row=[0, 2, 3, 0, 1], k=3 -> topk_values=[3, 2, 1], topk_indices=[2, 1, 4]
    values, indices = torch.topk(row, k)
    return values, indices

def print_confusion_report(confusion_matrix, class_index: int, k: int = 3):
    print("Confusion Matrix:")
    print(confusion_matrix)
    print("Total samples:", confusion_matrix.sum().item())
    #計算並印出每個類別的準確率
    class_accuracy = classwise_accuracy(confusion_matrix) * 100
    for index, name in enumerate(CLASS_NAMES):
        print(f"{name}: {class_accuracy[index].item():.2f}%")
    top_values, top_indices = get_top_confused_classes(
            confusion_matrix=confusion_matrix,
            class_index=class_index,
            k=k,
        )        
    #印出指定類別(class_index)被預測錯誤的前k個類別(top_indices)及其對應的樣本數(top_values)
    for value, index in zip(top_values, top_indices):
        print(
            f"{CLASS_NAMES[class_index]} -> {CLASS_NAMES[index.item()]}: "
            f"{value.item()} samples"
        )
