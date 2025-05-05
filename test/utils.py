import logging
import os
from datetime import datetime
from pathlib import Path

import pytest

from tabrel.utils.config import ClassifierConfig, ProjectConfig, TrainingConfig


def get_output_dir() -> Path:
    out_path_var_name = "OUT_PATH"
    result = Path(
        os.environ[out_path_var_name] if out_path_var_name in os.environ else "output"
    )
    result.mkdir(exist_ok=True, parents=True)
    return result


def make_test_dir(request: pytest.FixtureRequest) -> Path:
    result = (
        get_output_dir()
        / str(request.node.originalname)
        / datetime.now().isoformat(sep="_")
    )
    result.mkdir(parents=True)
    return result


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
