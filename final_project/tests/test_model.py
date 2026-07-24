import torch

from final_project.src.model import FashionCNN


def test_fashion_cnn_output_shape():
    model = FashionCNN()
    images = torch.rand(4, 1, 28, 28)
    #CNN模型的輸出應該是(batch_size, num_classes)，在這裡num_classes=10
    logits = model(images)

    assert logits.shape == (4, 10)  
    
def test_fashion_cnn_supports_custom_num_classes():
    #支援自定義類別數量的FashionCNN模型
    model = FashionCNN(num_classes=3)
    images = torch.rand(2, 1, 28, 28)

    logits = model(images)

    assert logits.shape == (2, 3)