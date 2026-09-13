import torch
from PIL import Image
from torch.utils.data import Dataset
from final_project_4.src.data import VOCDetectionDataset, VOCDetectionDataset, detection_collate_fn, parse_voc_annotation
# XML annotation for testing
raw_target = {
    "annotation": {
        "size": {
            "height": "100",
            "width": "160",
        },
        "object": [
            {
                "name": "cat",
                "difficult": "0",
                "bndbox": {
                    "xmin": "1",
                    "ymin": "1",
                    "xmax": "160",
                    "ymax": "100",
                },
            },
            {
                "name": "dog",
                "difficult": "1",
                "bndbox": {
                    "xmin": "21",
                    "ymin": "11",
                    "xmax": "60",
                    "ymax": "50",
                },
            },
        ],
    }
}

raw_target_empty = {
    "annotation": {
        "size": {
            "height": "100",
            "width": "160",
        }
    }
}



def test_parse_voc_annotation():
    image_id = 0
    parsed_annotation = parse_voc_annotation(raw_target, image_id)

    expected_boxes = torch.tensor([[0.0, 0.0, 160.0, 100.0], [20.0, 10.0, 60.0, 50.0]])
    expected_labels = torch.tensor([7, 11])  # cat -> 7, dog -> 11
    expected_difficults = torch.tensor([False, True])
    expected_image_id = torch.tensor(image_id, dtype=torch.int64)
    expected_orig_size = torch.tensor([100, 160], dtype=torch.int64)
    expected_size = expected_orig_size.clone()

    torch.testing.assert_close(parsed_annotation["boxes"], expected_boxes)
    torch.testing.assert_close(parsed_annotation["labels"], expected_labels)
    torch.testing.assert_close(parsed_annotation["difficult"], expected_difficults)
    torch.testing.assert_close(parsed_annotation["image_id"], expected_image_id)
    torch.testing.assert_close(parsed_annotation["orig_size"], expected_orig_size)
    torch.testing.assert_close(parsed_annotation["size"], expected_size)
    assert parsed_annotation["boxes"].dtype == torch.float32
    assert parsed_annotation["labels"].dtype == torch.int64
    assert parsed_annotation["difficult"].dtype == torch.bool
    assert parsed_annotation["image_id"].dtype == torch.int64
    assert parsed_annotation["orig_size"].dtype == torch.int64

def test_empty_annotation():
    image_id = 1
    parsed_annotation = parse_voc_annotation(raw_target_empty, image_id)

    expected_boxes = torch.empty((0, 4), dtype=torch.float32)
    expected_labels = torch.empty((0,), dtype=torch.int64)
    expected_difficults = torch.empty((0,), dtype=torch.bool)

    torch.testing.assert_close(parsed_annotation["boxes"], expected_boxes)
    torch.testing.assert_close(parsed_annotation["labels"], expected_labels)
    torch.testing.assert_close(parsed_annotation["difficult"], expected_difficults)

def test_detection_collate_fn():
    image_0 = torch.rand(3, 8, 10, dtype=torch.float32)
    image_1 = torch.rand(3, 8, 10, dtype=torch.float32)
    target_0 = {
        "boxes": torch.rand(1, 4, dtype=torch.float32)
    }
    target_1 = {
        "boxes": torch.rand(2, 4, dtype=torch.float32)
    }

    images, targets = detection_collate_fn([
    (image_0, target_0),
    (image_1, target_1),
    ])
    assert images.shape == torch.Size([2, 3, 8, 10])
    assert targets[0]["boxes"].shape == torch.Size([1, 4])
    assert targets[1]["boxes"].shape == torch.Size([2, 4])
    assert len(targets) == 2
    assert isinstance(images, torch.Tensor)
    assert isinstance(targets, list)
    
class FakeVOCDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        image = Image.new(
            "RGB",
            size=(200, 120),  # PIL: (W,H)
            color=(128, 64, 32),
        )

        raw_target = {
            "annotation": {
                "size": {
                    "height": "120",
                    "width": "200",
                },
                "object": [
                    {
                        "name": "dog",
                        "difficult": "1",
                        "bndbox": {
                            "xmin": "21",
                            "ymin": "11",
                            "xmax": "101",
                            "ymax": "81",
                        },
                    }
                ],
            }
        }

        return image, raw_target
    
def test_voc_detection_dataset():
    base_dataset = FakeVOCDataset()
    dataset = VOCDetectionDataset(
        base_dataset,
        output_size=(300, 400),
    )
    image, target = dataset[0]
    assert len(dataset) == 1
    assert image.shape == torch.Size([3, 300, 400])
    assert image.dtype == torch.float32
    assert isinstance(image, torch.Tensor)
    assert image.min() >= 0.0 and image.max() <= 1.0
    
    expected_boxes = torch.tensor(
        [[40.0, 25.0, 202.0, 202.5]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(target["boxes"], expected_boxes)
    assert target["boxes"].shape == torch.Size([1, 4])
    torch.testing.assert_close(target["labels"], torch.tensor([11], dtype=torch.int64))
    torch.testing.assert_close(target["difficult"], torch.tensor([True], dtype=torch.bool))
    assert target["image_id"].shape == torch.Size([])
    torch.testing.assert_close(target["image_id"], torch.tensor(0, dtype=torch.int64))
    torch.testing.assert_close(target["orig_size"], torch.tensor([120, 200], dtype=torch.int64))
    torch.testing.assert_close(target["size"], torch.tensor([300, 400], dtype=torch.int64))
    
    