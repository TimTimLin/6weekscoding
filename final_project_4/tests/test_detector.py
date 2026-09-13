import torch

from final_project_4.src.detector import FCOSDetector


def test_fcos_detector_forward():
    model = FCOSDetector(
        num_classes=20,
        fpn_channels=128,
        pretrained=False,
    )
    model.eval()

    images = torch.randn(2, 3, 320, 320)

    with torch.no_grad():
        output = model(images)

    expected_keys = {
        "class_logits",
        "box_regression",
        "centerness_logits",
        "locations",
        "strides",
    }

    assert isinstance(output, dict)
    assert set(output.keys()) == expected_keys

    # 40*40 + 20*20 + 10*10 = 2100
    assert output["class_logits"].shape == (2, 2100, 20)
    assert output["box_regression"].shape == (2, 2100, 4)
    assert output["centerness_logits"].shape == (2, 2100)
    assert output["locations"].shape == (2100, 2)
    assert output["strides"].shape == (2100,)

    # 檢查三個 level 的串接邊界
    torch.testing.assert_close(
        output["locations"][0],
        torch.tensor([4.0, 4.0]),
    )
    torch.testing.assert_close(
        output["locations"][1599],
        torch.tensor([316.0, 316.0]),
    )
    torch.testing.assert_close(
        output["locations"][1600],
        torch.tensor([8.0, 8.0]),
    )
    torch.testing.assert_close(
        output["locations"][2000],
        torch.tensor([16.0, 16.0]),
    )

    assert torch.all(output["strides"][:1600] == 8)
    assert torch.all(output["strides"][1600:2000] == 16)
    assert torch.all(output["strides"][2000:] == 32)

    # softplus 後的 box regression 應為正值
    assert torch.all(output["box_regression"] > 0)
    
    
def test_fcos_detector_imagenet_normalization():
    model = FCOSDetector(
    num_classes=20,
    fpn_channels=128,
    pretrained=False,
    normalize_inputs=True,
)
    images = torch.tensor(
        [0.485, 0.456, 0.406],
    ).view(1, 3, 1, 1)
    normalized = model.normalize_images(images)

    assert normalized.shape == images.shape
    assert normalized.dtype == images.dtype

    torch.testing.assert_close(
        normalized,
        torch.zeros_like(images),
        atol=1e-6,
        rtol=0.0,
    )