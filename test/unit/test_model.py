import pytest
import torch

from tabrel.model import TabularTransformerClassifier
from tabrel.utils.config import ClassifierConfig


@pytest.mark.parametrize(
    "input_shape, expected_output_shape",
    [
        ((32, 10), (16, 2)),
        ((64, 20), (32, 2)),
    ],
)
def test_tabular_transformer_classifier(
    input_shape: tuple[int, int], expected_output_shape: tuple[int, int]
) -> None:
    sample_size, num_features = input_shape
    n_classes = 2

    config = ClassifierConfig(
        n_features=num_features,
        d_embedding=3,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        num_layers=2,
        num_classes=n_classes,
        activation="relu",
        batch_query_ratio=0.5,
        dropout=0.1,
    )
    model = TabularTransformerClassifier(config)

    dummy_x, dummy_y = torch.randn(sample_size, num_features), torch.randint(
        low=0, high=2, size=(sample_size,)
    )
    output = model(dummy_x, dummy_y)
    assert (
        output.shape == expected_output_shape
    ), f"Output shape should be {(sample_size, n_classes)}, but got {output.shape}"
