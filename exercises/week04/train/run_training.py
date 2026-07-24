import torch
import copy
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split

from exercises.week04.train.cnn_model import TinyCNN 
from exercises.week04.train.training import train_one_epoch
from exercises.week04.train.evaluate import evaluate
from exercises.week04.train.toy_dataset import make_stripe_dataset

def main():
    torch.manual_seed(42)
    images, labels = make_stripe_dataset(n_per_class=100, noise_std=0.0) #生成一個簡單的二分類數據集，包含100個樣本，每個樣本是一個32x32的圖像，並且有兩個類別（0和1）
    dataset = TensorDataset(images, labels) #將圖像和標籤打包成一個數據集
    model = TinyCNN(in_channels=1, num_classes=2)
    #模型預測：logits
    #正確答案：labels
    loss_fn = nn.CrossEntropyLoss() #比較logits與正確答案，計算損失值
    #new_parameter = old_parameter - learning_rate × gradient
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )  
    
    generator = torch.Generator().manual_seed(42)  
    #隨機數據集切分的隨機種子，確保每次運行結果一致
    train_subset, val_subset = random_split(dataset, [160, 40], generator=generator) 
    #切分數據集，80%用於訓練，20%用於驗證
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, drop_last=False)
    #每個batchsize=16，將數據集分成多個批次進行訓練，最後一個批次可能小於16，drop_last=False表示保留最後一個批次，即使它的大小小於16
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, drop_last=False)
    #每個batchsize=16，將數據集分成多個批次進行訓練，shuffle=False表示不打亂數據集
    #紀錄每個epoch的訓練損失、訓練準確率、驗證損失、驗證準確率
    history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    #train_one_epoch()函數：訓練模型一個epoch，返回該epoch的平均損失值
    for i in range(100):
        epoch_loss = train_one_epoch(model, train_loader, loss_fn, optimizer) 
        print(f"Epoch {i+1}, Loss: {epoch_loss:.4f}")
        train_loss, train_accuracy = evaluate(model,train_loader,loss_fn,)
        print(f"Train Accuracy: {train_accuracy:.2f}%")
        #驗證模型在驗證集上的表現，返回驗證集的平均損失值和準確率
        validation_loss, validation_accuracy = evaluate(model, val_loader, loss_fn)
        print(f"Validation Loss: {validation_loss:.4f}, Accuracy: {validation_accuracy:.2f}%")
        # 儲存歷史紀錄
        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(validation_loss)
        history["val_accuracy"].append(validation_accuracy)       
        # 更新最佳模型權重
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_epoch = i + 1
            best_state = copy.deepcopy(model.state_dict())

    print(sum(parameter.numel() for parameter in model.parameters()))
    model.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
if __name__ == "__main__":
    main()