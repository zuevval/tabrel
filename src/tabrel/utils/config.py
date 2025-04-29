from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifierConfig:
    d_model: int  # Embedding dimension
    num_layers: int  # Transformer layers
    num_classes: int  # number of classes (= output dimension)
    dim_feedforward: int
    activation: str  # e.g. "relu"
    dropout: float

    @staticmethod
    def default() -> "ClassifierConfig":
        return ClassifierConfig(
            d_model=64,
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
    log_dir: str

    @staticmethod
    def default() -> "TrainingConfig":
        return TrainingConfig(batch_size=32, lr=1e-4, n_epochs=100, log_dir="logs")


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
