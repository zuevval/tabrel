import pytest
import torch

from tabrel.model import TabularTransformerClassifier
from tabrel.utils.config import ClassifierConfig


@pytest.mark.parametrize(
    "batch_size, query_size, num_features, expected_output_shape",
    [
        (32, 16, 10, (16, 2)),
        (64, 64, 20, (64, 2)),
    ],
)
def test_tabular_transformer_classifier(
    batch_size: int,
    query_size: int,
    num_features: int,
    expected_output_shape: tuple[int, int],
) -> None:
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
        dropout=0.1,
    )
    model = TabularTransformerClassifier(config)

    xb = torch.randn(batch_size, num_features)
    xq = torch.randn(query_size, num_features)
    yb = torch.randint(low=0, high=2, size=(batch_size,))
    output = model(xb, yb=yb, xq=xq)
    assert output.shape == expected_output_shape
