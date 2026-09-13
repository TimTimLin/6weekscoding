import torch
import torch.nn.functional as F

from final_project_5.src.model import DoubleConv, UNetEncoder , UNetDecoder, TinyUNet



def test_double_conv_preserves_spatial_shape():
    # 測試 DoubleConv 是否保持輸入的空間形狀
    model = DoubleConv(in_channels=3, out_channels=32)

    # 建立一個隨機的輸入張量
    x = torch.randn(2,3,16,24, dtype=torch.float32)

    # 前向傳播
    output = model(x)

    assert output.shape == (2, 32, 16, 24)
    assert output.dtype == torch.float32
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"
    assert torch.all(output >= 0), "Output values should be large than or equal to 0 due to ReLU activation"
    
def test_unet_encoder_shapes():
    model = UNetEncoder(in_channels=3, base_channels=32)
    x = torch.randn(2, 3, 64, 96, dtype=torch.float32)  # Batch size of 2, 3 channels, 64x96 image
    bottleneck_out, skip3, skip2, skip1 = model(x)
    assert bottleneck_out.shape == (2, 256, 8, 12)
    assert skip3.shape == (2, 128, 16, 24)
    assert skip2.shape == (2, 64, 32, 48)
    assert skip1.shape == (2, 32, 64, 96)  
    assert bottleneck_out.dtype == torch.float32
    assert skip3.dtype == torch.float32
    assert skip2.dtype == torch.float32
    assert skip1.dtype == torch.float32
    
def test_unet_decoder_shapes():
    model = UNetDecoder(base_channels=32)
    bottleneck_out = torch.randn(2, 256, 8, 12, dtype=torch.float32)
    skip3 = torch.randn(2, 128, 16, 24, dtype=torch.float32)
    skip2 = torch.randn(2, 64, 32, 48, dtype=torch.float32)
    skip1 = torch.randn(2, 32, 64, 96, dtype=torch.float32)
    output = model(bottleneck_out, skip3, skip2, skip1)
    assert output.shape == (2, 32, 64, 96)  
    assert output.dtype == torch.float32
    assert torch.all(output >= 0), "Output values should be large than or equal to 0 due to ReLU activation"
    
def test_tiny_unet_forward_and_backward():
    image = torch.randn(2, 3, 64, 96, dtype=torch.float32)  # Batch size of 2, 3 channels, 64x96 image
    mask = torch.randint(low=0, high=3, size=(2, 64, 96), dtype=torch.int64)  # Batch size of 2, 64x96 mask with values in {0,1,2}
    model = TinyUNet(in_channels=3, num_classes=3, base_channels=32)
    logits = model(image)
    predictions = torch.argmax(logits, dim=1)  # Get predicted class for each pixel (shape: [2, 64, 96])
    loss = F.cross_entropy(logits, mask)
    loss.backward()  # 應該不會出現錯誤
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
    assert logits.shape == (2, 3, 64, 96)  
    assert logits.dtype == torch.float32
    assert loss.ndim == 0  
    assert torch.isfinite(loss), "Loss should be finite"
    assert predictions.shape == (2, 64, 96)
    assert predictions.dtype == torch.int64
    
    
