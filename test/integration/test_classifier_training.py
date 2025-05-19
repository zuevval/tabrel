import logging
from pathlib import Path
from test.utils import light_config, make_test_dir
from typing import Final

import pytest
import torch

from tabrel.model import TabularTransformerClassifierModel
from tabrel.train import (
    generate_synthetic_data,
    load_checkpoint,
    save_checkpoint,
    train,
    wrap_data,
)
from tabrel.utils.logging import init_logging


def test_load_checkpoint(request: pytest.FixtureRequest) -> None:
    out_dir: Final[Path] = make_test_dir(request)
    num_features: Final[int] = 1
    num_classes: Final[int] = 2

    config = light_config(
        out_dir=out_dir, num_features=num_features, num_classes=num_classes
    )
    init_logging(config.training)
    model = TabularTransformerClassifierModel(config.model)
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

    x_train, y_train, r_train = generate_synthetic_data(
        num_samples=800, num_features=num_features, num_classes=num_classes
    )
    x_val, y_val, r_val = generate_synthetic_data(
        num_samples=200, num_features=num_features, num_classes=num_classes
    )

    out_dir = make_test_dir(request)
    config = light_config(
        out_dir=out_dir, num_features=num_features, num_classes=num_classes
    )

    train_dataset = wrap_data(x_train, y=y_train, r=r_train, config=config.training)
    val_dataset = wrap_data(x_val, y=y_val, r=r_val, config=config.training)

    init_logging(config.training)
    train(train_data=train_dataset, val_data=val_dataset, config=config)
