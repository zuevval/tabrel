from typing import Final

import torch
import torch.nn as nn
from utils.config import ClassifierConfig


class TabularTransformerClassifier(nn.Module):
    config: Final[ClassifierConfig]

    def __init__(self, config: ClassifierConfig):
        super(TabularTransformerClassifier, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=1,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,  # Important for tabular data (B, S, E) layout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.output_layer = nn.Linear(config.d_model, num_classes)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, self.config.d_model)
        Returns:
            logits: Tensor of shape (batch_size, self.config.num_classes)
        """

        # Expand dimensions: Transformer expects (B, S, E)
        x = x.unsqueeze(1)  # (B, 1, d_model)
        x = self.transformer_encoder(x)  # (B, 1, d_model)
        x = x.squeeze(1)  # (B, d_model)
        return self.output_layer(x)  # (B, num_classes) - logits


# Example Usage
if __name__ == "__main__":
    batch_size = 32
    num_features = 10
    num_classes = 2

    model = TabularTransformerClassifier(
        ClassifierConfig(
            d_model=num_features,
            num_layers=2,
            num_classes=num_classes,
            dim_feedforward=128,
            activation="relu",
            dropout=0.1,
        )
    )

    dummy_input = torch.randn(batch_size, num_features)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Should be (batch_size, num_classes)
