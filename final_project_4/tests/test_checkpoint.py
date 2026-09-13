import torch
import pytest
from final_project_4.src.checkpoint import save_checkpoint, load_checkpoint, build_checkpoint
from final_project_4.src.detector import FCOSDetector

def test_checkpoint_roundtrip(tmp_path):
    # 建立一個簡單的 checkpoint
    model_state = {
    "weight": torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        ]),
    }
    checkpoint = build_checkpoint(
        experiment="test_experiment",
        model_config={"param1": 42, "param2": "value"},
        model_state=model_state,
        optimizer_state={"lr": 0.001},
        epoch=5,
        step=100,
        metrics={"accuracy": 0.95},
        data={"extra_info": "test_data"},
    )
    # 儲存 checkpoint
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, checkpoint_path)   
    assert checkpoint_path.exists()
    # 載入 checkpoint
    loaded_checkpoint = load_checkpoint(checkpoint_path)
    assert loaded_checkpoint["format_version"] == 1
    assert loaded_checkpoint["experiment"] == "test_experiment"
    assert loaded_checkpoint["progress"] == {
        "epoch": 5,
        "step": 100,
    }
    assert loaded_checkpoint["model_config"] == {
        "param1": 42,
        "param2": "value",
    }
    assert loaded_checkpoint["optimizer_state"] == {"lr": 0.001}
    assert loaded_checkpoint["metrics"] == {"accuracy": 0.95}
    assert loaded_checkpoint["data"] == {"extra_info": "test_data"}
    torch.testing.assert_close(
        loaded_checkpoint["model_state"]["weight"],
        model_state["weight"],
    )

def test_load_checkpoint_rejects_missing_keys(tmp_path):
    # 建立一個缺少必要欄位的 checkpoint
    incomplete_checkpoint = {
        "format_version": 1,
        "experiment": "test_experiment",
        # 缺少 model_config, model_state, optimizer_state, progress, metrics, data
    }
    checkpoint_path = tmp_path / "incomplete_checkpoint.pt"
    save_checkpoint(incomplete_checkpoint, checkpoint_path)
    with pytest.raises(
        ValueError,
        match="missing required keys",
    ):
        load_checkpoint(checkpoint_path)
        
def test_checkpoint_reconstructs_detector():
    model_config = {
        "num_classes": 3,
        "fpn_channels": 32,
        "normalize_inputs": False,
    }
    original_model = FCOSDetector(**model_config)

    checkpoint = build_checkpoint(
        experiment="model_reconstruction_test",
        model_config=model_config,
        model_state=original_model.state_dict(),
    )

    restored_model = FCOSDetector(
        **checkpoint["model_config"]
    )

    restored_model.load_state_dict(
        checkpoint["model_state"]
    )
    torch.testing.assert_close(
    restored_model.head.class_logits.weight,
    original_model.head.class_logits.weight,
)