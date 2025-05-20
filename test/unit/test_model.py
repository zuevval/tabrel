from typing import Final

import pytest
import torch

from tabrel.model import (
    RelationalMultiheadAttention,
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


@pytest.mark.parametrize("embed_dim, num_heads", [(0, 1), (1, 0)])
def test_rel_mha_zero_params(embed_dim: int, num_heads: int) -> None:
    with pytest.raises(ValueError, match="`embed_dim` and `num_heads` must be >= 0"):
        RelationalMultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, rel=False
        )


def test_embed_dim_not_divisible_by_num_heads() -> None:
    with pytest.raises(
        ValueError, match="`embed_dim` must be divisible by `num_heads`"
    ):
        RelationalMultiheadAttention(embed_dim=10, num_heads=3, dropout=0.1, rel=False)


def test_rel_mha_properties() -> None:
    attn = RelationalMultiheadAttention(
        embed_dim=8, num_heads=2, dropout=0.1, rel=False
    )
    assert attn.head_dim == 4
    assert pytest.approx(attn.scaling_factor) == 4**-0.5


def test_rel_mha_forward_shape_mismatch() -> None:
    model = RelationalMultiheadAttention(
        embed_dim=8, num_heads=2, dropout=0.1, rel=False
    )
    x = torch.randn(5, 8)
    r = torch.randn(5, 5)
    sample = SampleWithRelations(x=x, r=r)
    attn_mask = torch.ones(6, 6, dtype=torch.bool)  # wrong shape
    with pytest.raises(ValueError, match="`attn_mask.shape` must be"):
        model(sample, attn_mask)


def test_rel_mha_forward_embed_dim_mismatch() -> None:
    model = RelationalMultiheadAttention(
        embed_dim=8, num_heads=2, dropout=0.1, rel=False
    )
    x = torch.randn(5, 6)  # wrong embed dim
    r = torch.randn(5, 5)
    sample = SampleWithRelations(x=x, r=r)
    attn_mask = torch.ones(5, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="x.shape\\[1\\] != embed_dim"):
        model(sample, attn_mask)


@pytest.mark.parametrize("use_rel", [False, True])
def test_rel_mha_forward_output_shape(use_rel: bool) -> None:
    n_samples: Final[int] = 5
    d_model: Final[int] = 8
    model = RelationalMultiheadAttention(
        embed_dim=d_model, num_heads=2, dropout=0.0, rel=use_rel
    )
    x = torch.randn(n_samples, d_model)
    r = torch.randn(n_samples, n_samples)
    sample = SampleWithRelations(x=x, r=r)
    attn_mask = torch.zeros(n_samples, n_samples, dtype=torch.bool)
    output = model(sample, attn_mask)
    assert output.shape == (n_samples, d_model)
    assert isinstance(output, torch.Tensor)


@pytest.fixture
def layer() -> RelTransformerEncoderLayer:
    return RelTransformerEncoderLayer(
        d_model=16,
        nhead=4,
        dim_feedforward=64,
        dropout=0.1,
        activation="relu",
        rel=True,
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
        RelTransformerEncoderLayer(16, 4, 64, 0.1, activation="invalid", rel=True)

    gelu_layer = RelTransformerEncoderLayer(16, 4, 64, 0.1, activation="gelu", rel=True)
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
        rel=True,
        dropout=0.1,
    )
    model = TabularTransformerClassifierModel(config)

    xb = torch.randn(batch_size, num_features)
    xq = torch.randn(query_size, num_features)
    yb = torch.randint(low=0, high=2, size=(batch_size,))
    r = torch.eye(batch_size + query_size)
    output = model(xb, yb=yb, xq=xq, r=r)
    assert output.shape == expected_output_shape
