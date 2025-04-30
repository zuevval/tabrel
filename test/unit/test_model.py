import pytest
import torch

from tabrel.model import TabularTransformerClassifier
from tabrel.utils.config import ClassifierConfig


@pytest.mark.parametrize(
    "input_shape, expected_output_shape",
    [
        ((32, 10), (32, 2)),
        ((64, 20), (64, 2)),
    ],
)
def test_tabular_transformer_classifier(
    input_shape: tuple[int, int], expected_output_shape: tuple[int, int]
) -> None:
    batch_size, num_features = input_shape
    n_classes = 2

    config = ClassifierConfig(
        n_features=num_features,
        d_embedding=3,
        num_layers=2,
        num_classes=n_classes,
        dim_feedforward=128,
        activation="relu",
        dropout=0.1,
    )
    model = TabularTransformerClassifier(config)

    dummy_input = torch.randn(batch_size, num_features)
    output = model(dummy_input)
    assert (
        output.shape == expected_output_shape
    ), f"Output shape should be {(batch_size, n_classes)}, but got {output.shape}"
