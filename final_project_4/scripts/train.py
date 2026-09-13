from pathlib import Path
    
import torch
from torch.utils.data import DataLoader

from final_project_4.src.data import (
    build_voc_dataset,
    detection_collate_fn,
)
from final_project_4.src.detector import FCOSDetector
from final_project_4.src.training import evaluate_one_epoch, train_one_epoch
from final_project_4.src.checkpoint import (
    build_checkpoint,
    save_checkpoint,
)

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"

    device = torch.device("cpu")

    train_dataset = build_voc_dataset(
        root=data_root,
        image_set="train",
        output_size=(320, 320),
        download=False,
    )
    val_dataset = build_voc_dataset(
        root=data_root,
        image_set="val",
        output_size=(320, 320),
        download=False,
    )   
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )
    # model_config 用於訓練模型和建立 checkpoint，方便在不同實驗中使用相同的模型設定，避免在不同地方手動維護兩份相同的資訊  
    model_config = {
        "num_classes": 20,
        "fpn_channels": 128,
        "normalize_inputs": True,
    }

    model = FCOSDetector(**model_config, pretrained=True).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    regression_ranges = {
        8: (0.0, 64.0),
        16: (64.0, 128.0),
        32: (128.0, float("inf")),
    }
    
    num_epochs = 5
    train_max_steps = None
    validation_max_steps = 100
    best_val_loss = float("inf")

    checkpoint_dir = project_root / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        checkpoint_dir / "pretrained_giou_longer.pt"
    )
    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            regression_ranges=regression_ranges,
            num_classes=20,
            max_steps=train_max_steps,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            dataloader=val_dataloader,
            device=device,
            regression_ranges=regression_ranges,
            num_classes=20,
            max_steps=validation_max_steps,
        )

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  train loss: {train_metrics['loss']:.4f}")
        print(f"  val loss:   {val_metrics['loss']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            checkpoint = build_checkpoint(
                experiment="pretrained_giou_longer",
                model_config=model_config,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                epoch=epoch + 1,
                metrics={
                    "train": train_metrics,
                    "validation": val_metrics,
                },
                data={
                    "dataset": "VOC2007",
                    "train_image_set": "train",
                    "validation_image_set": "val",
                    "output_size": (320, 320),
                    "regression_ranges": regression_ranges,
                },
            )

            save_checkpoint(checkpoint, checkpoint_path)
            print(f"  saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
