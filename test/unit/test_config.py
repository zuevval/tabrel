import numpy as np

from tabrel.utils.config import (
    ClassifierConfig,
    ProjectConfig,
    TrainingConfig,
)


# --- Class-Based Config Tests ---
def test_model_config() -> None:
    cfg = ClassifierConfig(
        d_model=64,
        num_layers=1,
        num_classes=2,
        dim_feedforward=128,
        activation="relu",
        dropout=0.0,
    )
    assert cfg.d_model == 64
    assert cfg.num_layers == 1
    assert np.isclose(cfg.dropout, 0.0)


def test_training_config() -> None:
    cfg = TrainingConfig.default()
    assert cfg.batch_size == 32
    assert np.isclose(cfg.lr, 1e-4)


def test_project_config() -> None:
    project_cfg = ProjectConfig.default()
    assert project_cfg.model.d_model == 64
    assert project_cfg.training.batch_size == 32
