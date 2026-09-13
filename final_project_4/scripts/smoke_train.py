from pathlib import Path

import torch
from torch.utils.data import DataLoader

from final_project_4.src.data import (
    build_voc_dataset,
    detection_collate_fn,
)
from final_project_4.src.detector import FCOSDetector
from final_project_4.src.training import train_one_epoch


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"

    device = torch.device("cpu")

    dataset = build_voc_dataset(
        root=data_root,
        image_set="train",
        output_size=(320, 320),
        download=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    model = FCOSDetector(
        num_classes=20,
        fpn_channels=128,
        pretrained=True,
        normalize_inputs=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    regression_ranges = {
        8: (0.0, 64.0),
        16: (64.0, 128.0),
        32: (128.0, float("inf")),
    }

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        regression_ranges=regression_ranges,
        num_classes=20,
        max_steps=2,
    )
    
    print(f"dataset size: {len(dataset)}")
    print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()