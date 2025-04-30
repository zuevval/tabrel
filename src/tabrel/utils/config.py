import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassifierConfig:
    n_features: int
    d_embedding: int  # Embedding dimension (for each feature)
    num_layers: int  # Transformer layers
    num_classes: int  # number of classes (= output dimension)
    dim_feedforward: int
    activation: str  # e.g. "relu"
    dropout: float

    @staticmethod
    def default() -> "ClassifierConfig":
        return ClassifierConfig(
            n_features=60,
            d_embedding=24,
            num_layers=3,
            num_classes=2,
            dim_feedforward=128,
            activation="relu",
            dropout=0.1,
        )


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    lr: float
    n_epochs: int
    log_dir: Path
    log_level: int
    print_logs_to_console: bool
    checkpoints_dir: Path
    allow_dirs_exist: bool

    def __post_init__(self) -> None:
        for dir in (self.log_dir, self.checkpoints_dir):
            dir.mkdir(parents=True, exist_ok=self.allow_dirs_exist)

    @staticmethod
    def default() -> "TrainingConfig":
        out_dir = Path("output")
        return TrainingConfig(
            batch_size=32,
            lr=1e-4,
            n_epochs=100,
            log_dir=out_dir / "logs",
            log_level=logging.INFO,
            print_logs_to_console=True,
            checkpoints_dir=out_dir / "checkpoints",
            allow_dirs_exist=True,
        )


@dataclass(frozen=True)
class ProjectConfig:
    model: ClassifierConfig
    training: TrainingConfig
    # use_wandb: bool    # TODO Enable Weights & Biases

    @staticmethod
    def default() -> "ProjectConfig":
        return ProjectConfig(
            model=ClassifierConfig.default(), training=TrainingConfig.default()
        )
