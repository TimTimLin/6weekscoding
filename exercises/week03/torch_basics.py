import torch 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
images = torch.rand(100, 3, 32, 32)
labels = torch.randint(0, 8, (100,))   
dataset = TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False)
batch_images, batch_labels = next(iter(dataloader))
print(len(dataset))
print(len(dataloader))
print(dataset[0][0].shape)
print(dataset[0][1])
print(batch_images.shape)
print(batch_labels.shape)
print(batch_images.dtype)
print(batch_images.device)