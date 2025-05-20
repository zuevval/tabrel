from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Final

import torch
import torch.nn as nn
from rtdl_num_embeddings import PeriodicEmbeddings  # type:ignore

from tabrel.utils.config import ClassifierConfig


# copied from PyTorch _get_activation_fn
def _get_activation_function(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return nn.functional.relu  # type:ignore
    elif activation == "gelu":
        return nn.functional.gelu  # type:ignore

    raise RuntimeError(f"activation should be gelu or relu, not {activation}")


@dataclass(frozen=True)
class SampleWithRelations:
    x: torch.Tensor
    r: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.x.shape) != 2 or len(self.r.shape) != 2:
            raise ValueError("x, r should be 2D tensors")
        n_samples: Final[int] = len(self.x)
        if self.r.shape != (n_samples, n_samples):
            raise ValueError(
                f"r.shape should be (n_samples, n_samples), "
                f"where n_samples=len(x)={n_samples},"
                f"but instead r.shape={self.r.shape}"
            )


class RelationalMultiheadAttention(nn.Module):
    """
    Based on a simplified torch.nn.MultiheadAttention
    (with additional relationship matrix)
    """

    embed_dim: Final[int]
    num_heads: Final[int]
    rel: Final[bool]  # use relationships matrix R or not

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def scaling_factor(self) -> float:
        return self.head_dim**-0.5  # type:ignore

    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float, rel: bool
    ) -> None:
        super().__init__()

        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("`embed_dim` and `num_heads` must be >= 0")

        if embed_dim % num_heads != 0:
            raise ValueError("`embed_dim` must be divisible by `num_heads`")

        self.embed_dim, self.num_heads, self.rel = embed_dim, num_heads, rel
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, s: SampleWithRelations, attn_mask: torch.Tensor) -> torch.Tensor:
        if attn_mask.shape != s.r.shape:
            raise ValueError(
                f"`attn_mask.shape` must be (n_samples, n_samples),"
                f"got {attn_mask.shape}"
            )
        n_samples, d_model = s.x.shape
        if d_model != self.embed_dim:
            raise ValueError(
                f"x.shape[1] != embed_dim: expected {self.embed_dim}, got {d_model}"
            )

        q, k, v = (
            (  # each tensor: (num_heads, n_samples, head_dim)
                proj(s.x)
                .reshape(n_samples, self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )

        # attention scores: (num_heads, n_samples, n_samples)
        attn_scores: torch.Tensor = (q @ k.transpose(-2, -1)) * self.scaling_factor

        if self.rel:
            attn_scores += s.r.unsqueeze(0)

        attn_scores = attn_scores.masked_fill(attn_mask == 0, -torch.inf)

        # attention weights: (num_heads, n_samples, n_samples)
        weights = torch.softmax(attn_scores, dim=-1)
        weights = self.dropout(weights)

        res = weights @ v  # (num_heads, n_samples, head_dim)
        res = res.transpose(0, 1).flatten(1)  # (n_samples, embed_dim)
        return self.out_proj(res)


class RelTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        rel: bool,
    ) -> None:
        super().__init__()
        self.self_attn = RelationalMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, rel=rel
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_function(activation)

    # a modified PyTorch _sa_block
    def _sa_block(
        self, s: SampleWithRelations, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.self_attn(s, attn_mask=attn_mask)
        return self.dropout1(x)  # type:ignore

    # a copy of original PyTorch _ff_block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)  # type:ignore

    # a modified PyTorch forward method
    def forward(self, src: SampleWithRelations, src_mask: torch.Tensor) -> torch.Tensor:
        src_mask = nn.functional._canonical_mask(  # type:ignore
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.x.dtype,
            check_other=False,
        )

        x = self.norm1(src.x + self._sa_block(src, src_mask))
        return self.norm2(x + self._ff_block(x))  # type:ignore


class RelTransformerEncoder(nn.Module):
    def __init__(
        self, encoder_layer: RelTransformerEncoderLayer, num_layers: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(self, src: SampleWithRelations, mask: torch.Tensor) -> torch.Tensor:
        output = src.x
        for layer in self.layers:
            input = SampleWithRelations(x=output, r=src.r)
            output = layer(input, src_mask=mask)
        return output


class TabularTransformerClassifierModel(nn.Module):
    config: Final[ClassifierConfig]
    d_input: Final[int]

    def __init__(self, config: ClassifierConfig):
        super(TabularTransformerClassifierModel, self).__init__()

        self.embeddings = PeriodicEmbeddings(
            n_features=config.n_features, d_embedding=config.d_embedding, lite=True
        )
        self.flatten = nn.Flatten()

        # project embedded features (+ y) to match Transformer input dimensions
        self.d_input = config.d_embedding * config.n_features + 1
        self.projection_layer = nn.Linear(self.d_input, config.d_model)

        encoder_layer = RelTransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            rel=config.rel,
        )
        self.transformer_encoder = RelTransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.output_layer: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(
            config.d_model, config.num_classes
        )
        self.config = config

    def forward(
        self,
        xb: torch.Tensor,
        yb: torch.Tensor,
        xq: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            xb: Tensor of shape (batch_size, self.config.n_features)
            yb: Tensor of shape (batch_size)
            xq: Tensor of shape (query_size, self.config.n_features)
            r: Tensor of shape (sample_size, sample_size) - relationships
                sample_size = batch_size + query_size; first batch, then query
        Returns:
            logits: Tensor of shape (query_size, self.config.num_classes)
        """
        x = torch.cat((xb, xq), 0)  # (S, config.n_features) [S = b+q - sample size]
        x = self.embeddings(x)  # (S, config.n_features, config.d_embedding)
        x = self.flatten(x)  # (S, d_input - 1)

        # add y
        sample_size, batch_size, query_size = len(x), len(xb), len(xq)
        y_masked = torch.cat((yb, torch.zeros(query_size)), 0)
        x = torch.cat((x, y_masked.unsqueeze(1)), 1)  # (S, d_input)
        x = self.projection_layer(x)  # (S, d_model)

        mask = torch.cat(
            (
                torch.zeros(sample_size, batch_size),
                torch.ones(sample_size, query_size) * -torch.inf,
            ),
            dim=1,
        )  # queries are not attended

        sample = SampleWithRelations(x=x, r=r)
        x = self.transformer_encoder(sample, mask=mask)  # (S, d_model)
        return self.output_layer(x)[batch_size:]  # (query_size, num_classes) - logits
