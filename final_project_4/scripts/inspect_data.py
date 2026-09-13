from pathlib import Path

import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import functional as TF

from final_project_4.src.data import (
    VOC_CLASSES,
    build_voc_dataset,
)
def main()-> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = PROJECT_ROOT / "data"
    OUTPUT_PATH = PROJECT_ROOT / "outputs" / "voc_sample.jpg"
    # 建立 VOCDetectionDataset 並下載 VOC 2007 train 資料集，並將影像 resize 成 (300, 400)
    dataset = build_voc_dataset(DATA_ROOT, image_set="train", output_size=(300, 400),download=True      )

    # 取得第一筆資料，並印出影像與標註資訊
    image, target = dataset[0]
    print(f"len(dataset): {len(dataset)}")
    print(f"Image shape: {image.shape}, dtype: {image.dtype}, range:{image.min().item()} to {image.max().item()}")
    print(f"box shape: {target['boxes'].shape}, dtype: {target['boxes'].dtype}, range:{target['boxes'].min().item()} to {target['boxes'].max().item()}")
    print(f"labels shape: {target['labels'].shape}, dtype: {target['labels'].dtype}")
    print(f"difficult shape: {target['difficult'].shape}, dtype: {target['difficult'].dtype}, ")
    print(f"orig_size {target['orig_size']}, size: {target['size']}, boxes: {target['boxes'][0]}")
    
    boxes = target["boxes"]
    labels = target["labels"]
    difficult_flags = target["difficult"]
    text_labels = []
    # 將標註資訊轉成文字標籤，若該物件為 difficult，則在標籤後面加上 "(difficult)" example: "dog(difficult)"
    for label,difficult in zip(labels, difficult_flags):
        # tensor(11) → 11 → VOC_CLASSES[11] → "dog"
        class_index = label.item()          
        class_name = VOC_CLASSES[class_index]
        if difficult.item()==True:
            text_labels.append(f"{class_name}(difficult)")
        else:
            text_labels.append(class_name)
    # 將image tensor轉成uint8格式，並將邊界框繪製在影像上
    image_uint8 = (image.clamp(0, 1)*255).round().to(torch.uint8)
    draw_image = draw_bounding_boxes(
        image_uint8,
        boxes,
        labels=text_labels,
        colors="red",
        width=2,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 將繪製好的影像轉成PIL image，並存檔
    pil_image = TF.to_pil_image(draw_image)
    pil_image.save(OUTPUT_PATH)
    print(f"Saved visualization to: {OUTPUT_PATH}")
if __name__ == "__main__":
    main()