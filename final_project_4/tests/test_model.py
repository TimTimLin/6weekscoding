
import torch
from torch.nn import functional as F
from final_project_4.src.model import ResNet18Backbone, FeaturePyramidNetwork,DetectionHead
    


def test_resnet18_backbone_output_shapes():
    model = ResNet18Backbone(pretrained=False)
    model.eval()  # Set the model to evaluation model
        
    image = torch.randn(2, 3, 320, 320)  # Batch of 2 images
    with torch.no_grad():  # Disable gradient calculation for inference    
        features = model(image)
    assert isinstance(features, dict)
    assert set(features.keys()) == {"c3", "c4", "c5"}
    assert features["c3"].shape == torch.Size([2, 128, 40, 40]) 
    assert features["c4"].shape == torch.Size([2, 256, 20, 20])
    assert features["c5"].shape == torch.Size([2, 512, 10, 10])
    assert features["c3"].dtype == image.dtype
    assert features["c4"].dtype == image.dtype
    assert features["c5"].dtype == image.dtype

def test_fpn_output_shapes_for_odd_feature_sizes():
    features = {
        "c3": torch.randn(2, 128, 38, 50),
        "c4": torch.randn(2, 256, 19, 25),
        "c5": torch.randn(2, 512, 10, 13),
    }

    fpn = FeaturePyramidNetwork(out_channels=96)
    fpn.eval()

    with torch.no_grad():
        pyramid = fpn(features)

    assert set(pyramid.keys()) == {"p3", "p4", "p5"}

    assert pyramid["p3"].shape == torch.Size([2, 96, 38, 50])
    assert pyramid["p4"].shape == torch.Size([2, 96, 19, 25])
    assert pyramid["p5"].shape == torch.Size([2, 96, 10, 13])

    assert pyramid["p3"].dtype == features["c3"].dtype
    assert pyramid["p4"].dtype == features["c4"].dtype
    assert pyramid["p5"].dtype == features["c5"].dtype

def test_detection_head_output_shapes():
    features = {
        "p3": torch.randn(2, 128, 32, 48),
        "p4": torch.randn(2, 128, 16, 24),
        "p5": torch.randn(2, 128, 8, 12),
    }

    head = DetectionHead(
        in_channels=128,
        num_classes=20,
    )
    head.eval()

    with torch.no_grad():
        predictions = head(features)

    assert predictions["p3"]["class_logits"].shape== torch.Size([2, 20, 32, 48])
    assert predictions["p3"]["box_regression"].shape == torch.Size([2, 4, 32, 48])
    assert predictions["p3"]["centerness_logits"].shape == torch.Size([2, 1, 32, 48])
    assert predictions["p4"]["class_logits"].shape == torch.Size([2, 20, 16, 24])
    assert predictions["p4"]["box_regression"].shape == torch.Size([2, 4, 16, 24])
    assert predictions["p4"]["centerness_logits"].shape == torch.Size([2, 1, 16, 24])
    assert predictions["p5"]["class_logits"].shape == torch.Size([2, 20, 8, 12])
    assert predictions["p5"]["box_regression"].shape == torch.Size([2, 4, 8, 12])
    assert predictions["p5"]["centerness_logits"].shape == torch.Size([2, 1, 8, 12])
    
    assert torch.all(
    predictions["p3"]["box_regression"] > 0
    ).item()
    
def test_detection_head_classification_prior():
    head = DetectionHead(
        in_channels=128,
        num_classes=20,
        prior_probability=0.01,
    )

    expected_probability = torch.full(
        (20,),
        0.01,
    )

    actual_probability = torch.sigmoid(
        head.class_logits.bias.detach()
    )

    torch.testing.assert_close(
        actual_probability,
        expected_probability,
        atol=1e-6,
        rtol=1e-6,
    )