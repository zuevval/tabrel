import logging
from datetime import datetime
from pathlib import Path

from torch.utils.data import TensorDataset

from tabrel.train import generate_synthetic_data, train
from tabrel.utils.config import ClassifierConfig, ProjectConfig, TrainingConfig
from tabrel.utils.logging import init_logging


def test_simple_training() -> None:
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

    out_dir = Path("output") / f"baseTransformer_syntheticData_{datetime.now()}"
    config = ProjectConfig(
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
    init_logging(config.training)
    train(train_data=train_dataset, val_data=val_dataset, config=config)
