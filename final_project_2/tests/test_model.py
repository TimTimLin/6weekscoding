import pytest
import torch

from final_project_2.src.model import CNN, CIFARResNet, CIFARPlainDeepNet

def test_baseline_cnn_output_shape():
    model = CNN()
    x = torch.randn(4, 3, 32, 32)

    logits = model(x)

    assert logits.shape == (4, 10)
def test_CIFARResNet_output_shape():
    model = CIFARResNet()
    x = torch.randn(4, 3, 32, 32)

    logits = model(x)

    assert logits.shape == (4, 10)
def test_CIFARPlainDeepNet_output_shape():
    model = CIFARPlainDeepNet()
    x = torch.randn(4, 3, 32, 32)

    logits = model(x)

    assert logits.shape == (4, 10)