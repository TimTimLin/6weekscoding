# 建立統一的 checkpoint 格式，方便儲存與載入模型、優化器狀態、訓練進度、評估指標與其他資料
# checkpoint
# ├── format_version: int
# ├── experiment: str
# ├── model_config: dict
# ├── model_state: dict[str, Tensor]
# ├── optimizer_state: dict | None
# ├── progress
# │   ├── epoch: int | None
# │   └── step: int | None
# ├── metrics: dict
# └── data: dict


from pathlib import Path
import torch


def build_checkpoint(
    *,
    experiment: str,
    model_config: dict[str, object],
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, object] | None = None,
    epoch: int | None = None,
    step: int | None = None,
    metrics: dict[str, object] | None = None,
    data: dict[str, object] | None = None,
) -> dict[str, object]:
    checkpoint = {
        "format_version": 1,
        "experiment": experiment,
        "model_config": model_config,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "progress": {
            "epoch": epoch,
            "step": step,
        },
        "metrics": {} if metrics is None else metrics,
        "data": {}  if data is None else data,
    }
    return checkpoint

def save_checkpoint(checkpoint: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

def load_checkpoint(path: str | Path, map_location: str |torch.device = "cpu") -> dict[str, object]:
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    required_keys = {
        "format_version",
        "experiment",
        "model_config",
        "model_state",
        "optimizer_state",
        "progress",
        "metrics",
        "data",
    }

    missing_keys = required_keys - checkpoint.keys()

    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required keys: {sorted(missing_keys)}"
        )

    if checkpoint["format_version"] != 1:
        raise ValueError(
            f"Unsupported checkpoint format version: "
            f"{checkpoint['format_version']}"
        )
    return checkpoint