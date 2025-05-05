import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassifierConfig:
    n_features: int
    d_embedding: int  # Embedding dimension (for each feature)
    d_model: int  # Transformer layer dimension (number of features)
    nhead: int  # Number of transformer heads. Required: d_model % nhead = 0
    dim_feedforward: int  # Feedforward model dimension of the Transformer layer
    num_layers: int  # Transformer layers
    num_classes: int  # number of classes (= output dimension)
    batch_query_ratio: float  # proportions of query samples in batch
    activation: str  # e.g. "relu"
    dropout: float

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError("`d_model` must be divisible by `nhead`")

    @staticmethod
    def default() -> "ClassifierConfig":
        return ClassifierConfig(
            n_features=60,
            d_embedding=24,
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            num_layers=3,
            num_classes=2,
            batch_query_ratio=0.3,
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
