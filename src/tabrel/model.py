from typing import Final

import torch
import torch.nn as nn
from rtdl_num_embeddings import PeriodicEmbeddings  # type:ignore

from tabrel.utils.config import ClassifierConfig


class TabularTransformerClassifier(nn.Module):
    config: Final[ClassifierConfig]
    d_model: Final[int]

    def __init__(self, config: ClassifierConfig):
        super(TabularTransformerClassifier, self).__init__()

        self.d_model = config.d_embedding * config.n_features
        self.embeddings = PeriodicEmbeddings(
            n_features=config.n_features, d_embedding=config.d_embedding, lite=True
        )
        self.flatten = nn.Flatten()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=1,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,  # Important for tabular data (B, S, E) layout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.output_layer = nn.Linear(self.d_model, config.num_classes)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, self.config.n_features)
        Returns:
            logits: Tensor of shape (batch_size, self.config.num_classes)
        """
        x = self.embeddings(x)  # (B, config.n_features, config.d_embedding)
        x = self.flatten(x)  # (B, d_model)

        # Expand dimensions: Transformer expects (B, S, E)
        x = x.unsqueeze(1)  # (B, 1, d_model)
        x = self.transformer_encoder(x)  # (B, 1, d_model)
        x = x.squeeze(1)  # (B, d_model)
        return self.output_layer(x)  # (B, num_classes) - logits
