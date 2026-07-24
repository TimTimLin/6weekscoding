import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split

from exercises.week03.train.simple_model import SimpleModel
from exercises.week03.train.training import train_one_epoch
from exercises.week03.train.evaluate import evaluate

def main():
    images = torch.rand(100, 3, 32, 32) #每張圖像的形狀為 (3, 32, 32)，100 是數據集大小
    labels = torch.randint(0, 10, (100,)) #每張圖像的標籤，隨機生成 0 到 9 的整數，表示 10 個類別
    dataset = TensorDataset(images, labels) #將圖像和標籤打包成一個數據集
    model = SimpleModel()
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
    train_subset, val_subset = random_split(dataset, [80, 20], generator=generator) 
    #切分數據集，80%用於訓練，20%用於驗證
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, drop_last=False)
    #每個batchsize=16，將數據集分成多個批次進行訓練，最後一個批次可能小於16，drop_last=False表示保留最後一個批次，即使它的大小小於16
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, drop_last=False)
    #每個batchsize=16，將數據集分成多個批次進行訓練，shuffle=False表示不打亂數據集
    #train_one_epoch()函數：訓練模型一個epoch，返回該epoch的平均損失值
    for i in range(10):
        epoch_loss = train_one_epoch(model, train_loader, loss_fn, optimizer) 
        print(f"Epoch {i+1}, Loss: {epoch_loss:.4f}")
        #validate the model after training for one epoch    
        validation_loss, validation_accuracy = evaluate(model, val_loader, loss_fn)
        print(f"Validation Loss: {validation_loss:.4f}, Accuracy: {validation_accuracy:.2f}%")

if __name__ == "__main__":
    main()