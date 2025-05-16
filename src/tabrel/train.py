import logging
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tabrel.dataset import QueryUniqueBatchDataset
from tabrel.model import TabularTransformerClassifierModel
from tabrel.utils.config import ProjectConfig, TrainingConfig
from tabrel.utils.linalg import mirror_triu


def run_epoch(
    model: TabularTransformerClassifierModel,
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
        device (torch.device): The device. Can be either a GPU or a CPU
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

    for xb, yb, xq, yq, r in dataloader:
        xb = xb.to(device)  # x_batch
        yb = yb.to(device)  # y_batch
        xq = xq.to(device)  # x_query
        yq = yq.to(device)  # y_query
        r = r.to(device)  # relationships

        if optimizer:  # if train mode
            optimizer.zero_grad()

        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(xb, yb, xq, r)

            loss = criterion(outputs, yq)

            if optimizer:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * len(xb)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == yq).sum().item()
        total += len(yq)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


_model_state_key: Final[str] = "model_state"
_optim_state_key: Final[str] = "optimizer_state"


def save_checkpoint(
    model: TabularTransformerClassifierModel,
    optimizer: torch.optim.Adam,
    output_path: Path,
    add_dict: dict[str, Any] | None = None,
) -> None:
    dict_to_save = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if add_dict:
        dict_to_save.update(add_dict)
    torch.save(dict_to_save, output_path)
    logging.info(f"checkpoint saved at {output_path}")


def load_checkpoint(
    config: ProjectConfig, device: torch.device, checkpoint_path: Path | None = None
) -> tuple[TabularTransformerClassifierModel, optim.Adam]:
    if not checkpoint_path:
        checkpoint_dir = Path(config.training.checkpoints_dir)
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if len(checkpoints) == 0:
            msg = f"no checkpoints in {checkpoint_dir}"
            logging.error(msg)
            raise FileNotFoundError(msg)
        checkpoint_path = checkpoints[-1]
    logging.info(f"Loading model and optimizer from checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)

    model = TabularTransformerClassifierModel(config.model).to(device)
    model.load_state_dict(state_dict[_model_state_key])
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
    optimizer.load_state_dict(state_dict[_optim_state_key])

    return model, optimizer


def train(
    train_data: QueryUniqueBatchDataset,
    val_data: QueryUniqueBatchDataset,
    config: ProjectConfig,
) -> None:
    train_loader = DataLoader(train_data, batch_size=None)
    val_loader = DataLoader(val_data, batch_size=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularTransformerClassifierModel(config.model).to(device)

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    r = torch.randint(0, 2, (num_samples, num_samples))
    return x, y, mirror_triu(r)


def wrap_data(
    x: torch.Tensor, y: torch.Tensor, r: torch.Tensor, config: TrainingConfig
) -> QueryUniqueBatchDataset:
    return QueryUniqueBatchDataset(
        x=x,
        y=y,
        r=r,
        query_size=config.query_size,
        batch_size=config.batch_size,
        n_batches=config.n_batches,
        random_state=config.random_seed,
    )
