from pathlib import Path

import torch
from final_project_4.src.checkpoint import (
    build_checkpoint,
    save_checkpoint,
)

from final_project_4.src.data import (
    build_voc_dataset,
)
from final_project_4.src.detector import FCOSDetector
from final_project_4.src.training import train_one_step

def main() -> None:
    data_root = Path(__file__).resolve().parents[1] / "data"

    dataset = build_voc_dataset(
        data_root,
        image_set="train",
        output_size=(320, 320),
        download=False,
    )

    for index in range(len(dataset)):
        image, target = dataset[index]
        if target["boxes"].shape[0] > 0:
            break

    model_config = {
        "num_classes": 20,
        "fpn_channels": 128,
        "normalize_inputs": True,
    }

    model = FCOSDetector(**model_config, pretrained=True).to(torch.device("cpu"))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    regression_ranges = {
        8: (0.0, 64.0),
        16: (64.0, 128.0),
        32: (128.0, float("inf")),
    }

    best_loss = float("inf")
    best_state = None
    best_step = None
    for step in range(100):
        losses = train_one_step(
            model=model,
            optimizer=optimizer,
            images=image.unsqueeze(0),
            dataset_targets=[target],
            regression_ranges=regression_ranges,
            num_classes=20,
        )
        loss_value = losses["loss"].item()

        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_step = step + 1
            
        if step % 10 == 0:
            print(step, losses["loss"].item())
    if best_state is None or best_step is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_state)

    checkpoint_path = (
        data_root.parent
        / "outputs"
        / "checkpoints"
        / "overfit_pretrained_smooth_l1.pt"
    )

    checkpoint = build_checkpoint(
        experiment="overfit_pretrained_smooth_l1",
        model_config=model_config,
        model_state=best_state,
        optimizer_state=None,
        step=best_step,
        metrics={
            "best_loss": best_loss,
        },
        data={
            "dataset": "VOC2007",
            "image_set": "train",
            "image_index": index,
            "output_size": (320, 320),
            "regression_ranges": regression_ranges,
        },
    )

    save_checkpoint(checkpoint, checkpoint_path)
    print(f"best loss: {best_loss:.4f}")
    print(f"saved checkpoint: {checkpoint_path}")
if __name__ == "__main__":
    main()