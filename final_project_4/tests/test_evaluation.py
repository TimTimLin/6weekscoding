import torch
from torch import nn

from final_project_4.src.evaluation import (
    evaluate_detection_model,
)


class FakePerfectDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_index = 0

    def forward(
        self,
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.call_index == 0:
            class_logits = images.new_tensor([
                [[10.0, -10.0]],
            ])
        else:
            class_logits = images.new_tensor([
                [[-10.0, 10.0]],
            ])

        self.call_index += 1

        return {
            "class_logits": class_logits,
            "box_regression": images.new_tensor([
                [[5.0, 5.0, 5.0, 5.0]],
            ]),
            "centerness_logits": images.new_tensor([
                [10.0],
            ]),
            "locations": images.new_tensor([
                [5.0, 5.0],
            ]),
        }
        
        
def test_evaluate_detection_model_across_batches():
    model = FakePerfectDetector()
    
    dataloader = [
        (
            torch.randn(1, 3, 10, 10),
            [
                {
                    "labels": torch.tensor([0], dtype=torch.int64),
                    "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                    "difficult": torch.tensor([False], dtype=torch.bool),
                },
            ],
        ),
        (
            torch.randn(1, 3, 10, 10),
            [
                {
                    "labels": torch.tensor([1], dtype=torch.int64),
                    "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                    "difficult": torch.tensor([False], dtype=torch.bool),
                },
            ],
        ),
    ]

    metrics = evaluate_detection_model(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        num_classes=2,
    )
    torch.testing.assert_close(
        metrics["ap_per_class"],
        torch.tensor([1.0, 1.0]),
    )

    torch.testing.assert_close(
        metrics["map"],
        torch.tensor(1.0),
    )

    torch.testing.assert_close(
        metrics["num_ground_truths_per_class"],
        torch.tensor([1, 1], dtype=torch.int64),
    )