from typing import Callable, Final

import torch
import torch.nn as nn
from rtdl_num_embeddings import PeriodicEmbeddings  # type:ignore

from tabrel.utils.config import ClassifierConfig


class TabularTransformerClassifier(nn.Module):
    config: Final[ClassifierConfig]
    d_input: Final[int]

    def __init__(self, config: ClassifierConfig):
        super(TabularTransformerClassifier, self).__init__()

        self.embeddings = PeriodicEmbeddings(
            n_features=config.n_features, d_embedding=config.d_embedding, lite=True
        )
        self.flatten = nn.Flatten()

        # project embedded features to match Transformer input dimensions
        self.d_input = config.d_embedding * config.n_features + 1
        self.projection_layer = nn.Linear(self.d_input, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,  # Important for tabular data (B, S, F) layout
        )  # B - batch size; S - sequence length; F - number of features (d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.output_layer: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(
            config.d_model, config.num_classes
        )
        self.config = config

    def forward(
        self, xb: torch.Tensor, yb: torch.Tensor, xq: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            xb: Tensor of shape (batch_size, self.config.n_features)
            yb: Tensor of shape (batch_size)
            xq: Tensor of shape (query_size, self.config.n_features)
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

        # Expand dimensions: Transformer expects (B, S, F)
        x = x.unsqueeze(0)  # (1, S, d_model)
        x = self.transformer_encoder(x, mask=mask)  # (1, S, d_model)
        x = x.squeeze(0)  # (S, d_model)
        return self.output_layer(x)[batch_size:]  # (query_size, num_classes) - logits
