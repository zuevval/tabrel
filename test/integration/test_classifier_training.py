import logging
from pathlib import Path
from test.utils import make_test_dir
from typing import Final

import pytest
import torch
from torch.utils.data import TensorDataset

from tabrel.model import TabularTransformerClassifier
from tabrel.train import (
    generate_synthetic_data,
    load_checkpoint,
    save_checkpoint,
    train,
)
from tabrel.utils.config import ClassifierConfig, ProjectConfig, TrainingConfig
from tabrel.utils.logging import init_logging


def light_config(out_dir: Path, num_features: int, num_classes: int) -> ProjectConfig:
    return ProjectConfig(
        model=ClassifierConfig(
            n_features=num_features,
            d_embedding=4,
            d_model=50,
            nhead=2,
            dim_feedforward=128,
            num_layers=2,
            num_classes=num_classes,
            activation="relu",
            dropout=0.1,
        ),
        training=TrainingConfig(
            batch_size=32,
            lr=1e-3,
            n_epochs=4,
            log_dir=out_dir / "logs",
            log_level=logging.DEBUG,
            print_logs_to_console=False,
            checkpoints_dir=out_dir / "checkpoints",
            allow_dirs_exist=False,
        ),
    )


def test_load_checkpoint(request: pytest.FixtureRequest) -> None:
    out_dir: Final[Path] = make_test_dir(request)
    num_features: Final[int] = 1
    num_classes: Final[int] = 2

    config = light_config(
        out_dir=out_dir, num_features=num_features, num_classes=num_classes
    )
    init_logging(config.training)
    model = TabularTransformerClassifier(config.model)
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint_path = config.training.checkpoints_dir / "checkpoint.pth"
    save_checkpoint(model=model, optimizer=optimizer, output_path=checkpoint_path)
    logging.info(f"Checkpoint saved at {checkpoint_path}")

    # test loading checkpoint directly specifying its path
    device = torch.device("cpu")
    loaded_model, _ = load_checkpoint(
        config=config, device=device, checkpoint_path=checkpoint_path
    )
    assert loaded_model.config.n_features == config.model.n_features

    # test loading the latest checkpoint
    latest_model, _ = load_checkpoint(config=config, device=device)
    assert latest_model.config.n_features == config.model.n_features


def test_simple_training(request: pytest.FixtureRequest) -> None:
    num_features = 10
    num_classes = 2

    x_train, y_train = generate_synthetic_data(
        num_samples=800, num_features=num_features, num_classes=num_classes
    )
    x_val, y_val = generate_synthetic_data(
        num_samples=200, num_features=num_features, num_classes=num_classes
    )

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    out_dir = make_test_dir(request)
    config = light_config(
        out_dir=out_dir, num_features=num_features, num_classes=num_classes
    )
    init_logging(config.training)
    train(train_data=train_dataset, val_data=val_dataset, config=config)
