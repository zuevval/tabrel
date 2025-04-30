import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn
import torch.optim as optim
from model import TabularTransformerClassifier
from torch.utils.data import DataLoader, TensorDataset
from utils.config import ClassifierConfig, ProjectConfig, TrainingConfig
from utils.logging import init_logging


def run_epoch(
    model: TabularTransformerClassifier,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """
    Run a single epoch of training or evaluation on the given dataset.

    Args:
        model (TabularTransformerClassifier): The NN model
        dataloader (DataLoader): A data loader for the training or validation set
        criterion (torch.nn.Module): The loss function
        device (torch.device): The device. Can be either a GPU or a CPU.
        optimizer (torch.optim.Optimizer | None, optional): If specified,
            the optimizer for gradient descent.
            If None, the model is not trained during the current epoch

    Returns:
        tuple[float, float]: A tuple containing the average loss
        and the accuracy on the validation set
    """
    if optimizer:  # if train mode
        model.train()
    else:
        model.eval()

    running_loss: float = 0.0
    correct: int = 0
    total: int = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if optimizer:  # if train mode
            optimizer.zero_grad()

        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            if optimizer:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * x_batch.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


_model_state_key: Final[str] = "model_state"
_optim_state_key: Final[str] = "optimizer_state"


def save_checkpoint(
    model: TabularTransformerClassifier,
    optimizer: torch.optim.Adam,
    output_path: Path,
    add_dict: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            **add_dict,
        },
        output_path,
    )
    logging.info(f"checkpoint saved at {output_path}")


def load_checkpoint(
    config: ProjectConfig, device: torch.device, checkpoint_path: Path | None = None
) -> tuple[TabularTransformerClassifier, optim.Adam]:
    if not checkpoint_path:
        checkpoint_dir = Path(config.training.checkpoints_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory '{checkpoint_dir}' "
                "does not exist and `checkpoint_path` not set"
            )

    latest_checkpoint = list(checkpoint_dir.glob("*.pth"))[-1]
    logging.info(f"Loading model and optimizer from checkpoint: {latest_checkpoint}")

    state_dict = torch.load(latest_checkpoint, map_location=device)

    model = TabularTransformerClassifier(config.model).to(device)
    model.load_state_dict(state_dict[_model_state_key])
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
    optimizer.load_state_dict(state_dict[_optim_state_key])

    return model, optimizer


def train(
    train_data: TensorDataset, val_data: TensorDataset, config: ProjectConfig
) -> None:
    train_loader = DataLoader(
        train_data, batch_size=config.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_data, batch_size=config.training.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularTransformerClassifier(config.model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)

    for i_epoch in range(config.training.n_epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        logging.info(
            f"Epoch [{i_epoch+1}/{config.training.n_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            add_dict={
                "i_epoch": i_epoch,
                "train": {"loss": train_loss, "acc": train_acc},
                "val": {"loss": val_loss, "acc": val_acc},
            },
            output_path=config.training.checkpoints_dir / f"checkpoint_{i_epoch}.pth",
        )


# Create synthetic data (for testing purposes)
def generate_synthetic_data(
    num_samples: int, num_features: int, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return x, y


def main() -> None:
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
            d_model=num_features,
            num_layers=2,
            num_classes=num_classes,
            dim_feedforward=128,
            activation="relu",
            dropout=0.1,
        ),
        training=TrainingConfig(
            batch_size=32,
            lr=1e-3,
            n_epochs=10,
            log_dir=out_dir / "logs",
            checkpoints_dir=out_dir / "checkpoints",
            allow_dirs_exist=False,
        ),
    )
    init_logging(config.training, level=logging.DEBUG, print_logs=True)
    train(train_data=train_dataset, val_data=val_dataset, config=config)


if __name__ == "__main__":
    main()
