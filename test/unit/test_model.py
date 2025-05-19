import pytest
import torch

from tabrel.model import (
    RelTransformerEncoderLayer,
    SampleWithRelations,
    TabularTransformerClassifierModel,
)
from tabrel.utils.config import ClassifierConfig


def test_sample_with_relations() -> None:
    # Valid input
    s = SampleWithRelations(x=torch.randn(4, 3), r=torch.randn(4, 4))
    assert s.x.shape == (4, 3)
    assert s.r.shape == (4, 4)

    # Invalid: x is not 2D
    with pytest.raises(ValueError, match="x, r should be 2D tensors"):
        SampleWithRelations(x=torch.randn(4), r=torch.randn(4, 4))

    # Invalid: r is not 2D
    with pytest.raises(ValueError, match="x, r should be 2D tensors"):
        SampleWithRelations(torch.randn(4, 3), torch.randn(4))

    # Invalid: x, r dimensions disagree
    with pytest.raises(ValueError, match="r.shape should be"):
        SampleWithRelations(torch.randn(4, 3), torch.randn(4, 3))


@pytest.fixture
def layer() -> RelTransformerEncoderLayer:
    return RelTransformerEncoderLayer(
        d_model=16, nhead=4, dim_feedforward=64, dropout=0.1, activation="relu"
    )


def make_sample(
    n_samples: int, d_model: int
) -> tuple[SampleWithRelations, torch.Tensor]:
    x = torch.randn(n_samples, d_model)
    r = torch.eye(n_samples)  # simple identity relation matrix
    src_mask = torch.zeros(n_samples, n_samples, dtype=torch.bool)
    return SampleWithRelations(x=x, r=r), src_mask


def test_forward_output_shape(layer: pytest.fixture) -> None:
    n_samples, d_model = 8, 16
    sample, src_mask = make_sample(n_samples=n_samples, d_model=d_model)
    output = layer(sample, src_mask)
    assert output.shape == (n_samples, d_model)


def test_forward_requires_grad(layer: pytest.fixture) -> None:
    n_samples, d_model = 8, 16
    sample, src_mask = make_sample(n_samples=n_samples, d_model=d_model)
    sample.x.requires_grad = True
    output = layer(sample, src_mask)
    loss = output.sum()
    loss.backward()
    assert sample.x.grad is not None


def test_activations() -> None:
    with pytest.raises(RuntimeError, match="activation should be gelu or relu"):
        RelTransformerEncoderLayer(16, 4, 64, 0.1, activation="invalid")

    gelu_layer = RelTransformerEncoderLayer(16, 4, 64, 0.1, activation="gelu")
    assert gelu_layer.activation is torch.nn.functional.gelu


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
    model = TabularTransformerClassifierModel(config)

    xb = torch.randn(batch_size, num_features)
    xq = torch.randn(query_size, num_features)
    yb = torch.randint(low=0, high=2, size=(batch_size,))
    r = torch.eye(batch_size + query_size)
    output = model(xb, yb=yb, xq=xq, r=r)
    assert output.shape == expected_output_shape
