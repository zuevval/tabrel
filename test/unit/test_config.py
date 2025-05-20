import dataclasses

import numpy as np
import pytest

from tabrel.utils.config import (
    ClassifierConfig,
    ProjectConfig,
    TrainingConfig,
)


# --- Class-Based Config Tests ---
def test_classifier_config_basic() -> None:
    cfg = ClassifierConfig(
        n_features=10,
        d_embedding=25,
        d_model=10,
        nhead=2,
        dim_feedforward=128,
        num_layers=1,
        num_classes=2,
        activation="relu",
        rel=True,
        dropout=0.0,
    )
    assert cfg.d_embedding == 25
    assert cfg.num_layers == 1
    assert np.isclose(cfg.dropout, 0.0)


def test_classifier_config_exception() -> None:
    cfg = ClassifierConfig.default()
    assert cfg.d_model == 64

    with pytest.raises(ValueError):
        dataclasses.replace(cfg, nhead=5)


def test_training_config() -> None:
    cfg = TrainingConfig.default()
    assert cfg.batch_size == 32
    assert np.isclose(cfg.lr, 1e-4)
    assert cfg.checkpoints_dir.exists()


def test_project_config() -> None:
    project_cfg = ProjectConfig.default()
    assert project_cfg.model.d_embedding == 24
    assert project_cfg.training.batch_size == 32
