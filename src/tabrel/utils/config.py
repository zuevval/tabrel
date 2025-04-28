from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:  # TODO use Pydantic
    d_model: int  # Embedding dimension
    nhead: int  # Attention heads
    num_layers: int  # Transformer layers
    dropout: float

    @staticmethod
    def default() -> "ModelConfig":
        return ModelConfig(d_model=64, nhead=4, num_layers=3, dropout=0.1)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    lr: float
    epochs: int
    log_dir: str

    @staticmethod
    def default() -> "TrainingConfig":
        return TrainingConfig(batch_size=32, lr=1e-4, epochs=100, log_dir="logs")


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig
    training: TrainingConfig
    # use_wandb: bool    # TODO Enable Weights & Biases

    @staticmethod
    def default() -> "ProjectConfig":
        return ProjectConfig(
            model=ModelConfig.default(), training=TrainingConfig.default()
        )
