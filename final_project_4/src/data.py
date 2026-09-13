import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from pathlib import Path
from torchvision.datasets import VOCDetection
from final_project_4.src.transforms import resize_image_and_boxes
"""
VOC XML
   ↓
name = "dog"
   ↓ VOC_CLASS_TO_IDX["dog"]
label = 11
   ↓
Tensor[N] int64
"""
#建立VOC_CLASSES index -> class name 
VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
#建立VOC_CLASS_TO_IDX class name -> index
VOC_CLASS_TO_IDX = {
    name: index
    for index, name in enumerate(VOC_CLASSES)
}
#把XML annotation轉成tensor格式
def parse_voc_annotation(raw_target: dict,image_id:int) -> dict[str,torch.Tensor]:
    #從annotation中取得height, width, objects
    annotation = raw_target["annotation"]
    height = int(annotation["size"]["height"])
    width = int(annotation["size"]["width"])
    # 如果objects為空，則返回空的tensor
    objects = annotation.get("object", [])
    labels = []
    difficults = []
    boxes = []
    for obj in objects:
        name= obj["name"]
        #將class name轉換成label index ex dog -> 11 
        label = VOC_CLASS_TO_IDX[name]
        difficult = bool(int(obj["difficult"]))
        # VOC 使用 1-based pixel 座標，且 xmax/ymax 包含最後一個 pixel。
        # 轉成專案內部的 0-based continuous xyxy 邊界：
        # xmin/ymin 需 -1；xmax/ymax 保持不變。
        xmin = float(obj["bndbox"]["xmin"])-1.0
        ymin = float(obj["bndbox"]["ymin"])-1.0
        xmax = float(obj["bndbox"]["xmax"])
        ymax = float(obj["bndbox"]["ymax"])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
        difficults.append(difficult)
    # reshape boxes 
    # N > 0 → [N,4]
    # N = 0 → [0,4]
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    #把list轉成tensor格式
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    difficults_tensor = torch.tensor(difficults, dtype=torch.bool)
    image_id_tensor = torch.tensor(image_id, dtype=torch.int64)
    orig_size = torch.tensor([height, width], dtype=torch.int64)
    return {
    "boxes": boxes_tensor,
    "labels": labels_tensor,
    "difficult": difficults_tensor,
    "image_id": image_id_tensor,
    "orig_size": orig_size,
    "size": orig_size.clone(),
    }

# input batch = [
#   (image_0, target_0),
#   (image_1, target_1),
#  ...]
# output = (
#   batched_images: Tensor[B, C, H, W],
#   [target_0, target_1, ...]
# )
def detection_collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    #zip 拆開input batch 變成 (image_0, image_1, ...), (target_0, target_1, ...)
    images, targets = list(zip(*batch))
    # 將image堆疊起來變(B, C, H, W)
    batched_images = torch.stack(images, dim=0)   
    targets = list(targets)  # targets is already a list of dictionaries
    return batched_images, targets


# VOCDetection[index]
#         │
#         ├──────────────┐
#         ▼              ▼
#    PIL.Image        raw XML
#         │              │
#   pil_to_tensor         │
#         │         parse_voc_annotation()
#         │              │
#         ▼              ▼
#  Tensor image       target dict
#         │              │
#         └──────┬───────┘
#                ▼
#      resize_image_and_boxes()
#                │
#                ▼
#         image + target
#                │
#                ▼
#          DataLoader


class VOCDetectionDataset(Dataset):
    def __init__(self, base_dataset: Dataset, output_size: tuple[int, int])-> None:
        self.base_dataset = base_dataset
        self.output_size = (output_size[0], output_size[1])  # (new_height, new_width)
    # 資料集的長度等於基礎資料集的長度 
    def __len__(self)-> int:
        return len(self.base_dataset)
    # 取得指定索引的資料，return (image[3, H, W], target[dict[str, torch.Tensor]])
    def __getitem__(self, idx:int)-> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        #取出第idx個PIL image和raw XML annotation
        pil_image, raw_target = self.base_dataset[idx]   
        # PIL RGB → float32 Tensor[3,H,W], range [0,1]
        image= TF.to_tensor(pil_image)  
        # 將XML annotation轉成tensor格式的target dict
        target = parse_voc_annotation(raw_target, idx)
        # 將image和target["boxes"] resize成指定大小，並更新target["boxes"]和target["size"]
        image,resized_boxes = resize_image_and_boxes(image, target["boxes"], self.output_size)
        target["boxes"] = resized_boxes
        target["size"] = torch.tensor(image.shape[-2:], dtype=torch.int64)
        return image, target


# 載入VOC資料集，並建立VOCDetectionDataset
def build_voc_dataset(root: str | Path, image_set: str, output_size: tuple[int, int] = (320, 320),
    download: bool = False) -> VOCDetectionDataset:
    if image_set not in ["train", "val", "test"]:
        raise ValueError(f"Invalid image_set: {image_set}. Must be one of 'train', 'val', or 'test'.")
    base_dataset = VOCDetection(
        root=root,
        year="2007",
        image_set=image_set,
        download=download,
    )
    return VOCDetectionDataset(base_dataset, output_size=output_size)