import logging
from pathlib import Path
from typing import Any, Final, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from rtdl_num_embeddings import PeriodicEmbeddings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabrel.dataset import QueryUniqueBatchDataset
from tabrel.model import RelMHARegressor, TabularTransformerClassifierModel
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
        backgnd_size=config.backgnd_size,
        n_batches=config.n_batches,
        random_state=config.random_seed,
    )


def train_relnet(
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    backgnd_indices: np.ndarray,
    query_indices: np.ndarray,
    val_indices: np.ndarray,
    lr: float,
    n_epochs: int,
    n_layers: int = 2,
    embed_dim: int = 32,
    periodic_embed_dim: int | None = 5,
    num_heads: int = 1,
    dropout: float = 0.1,
    progress_bar: bool = True,
    print_loss: bool = False,
    lr_decay: float | None = None,
    lr_decay_step: int = 100,
    tb_logdir: str | None = "tensorboard_logs",
) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    torch.random.seed
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    r_tensor = torch.tensor(r, dtype=torch.float32, device=device)

    # Masks
    n = x.shape[0]
    backgnd_mask = torch.zeros(n, dtype=torch.bool, device=device)
    backgnd_mask[backgnd_indices] = True

    probe_mask = torch.zeros(n, dtype=torch.bool, device=device)
    probe_mask[query_indices] = True

    val_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask[val_indices] = True

    # Model & optimizer
    if periodic_embed_dim is not None:
        embeddings = PeriodicEmbeddings(
            n_features=x_tensor.shape[1], d_embedding=periodic_embed_dim, lite=True
        )
        flatten = nn.Flatten()
        in_dim = periodic_embed_dim * x.shape[1] + 1
    else:
        in_dim = x.shape[1] + 1

    model = RelMHARegressor(
        in_dim=in_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        rel=True,
        num_layers=n_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    scheduler = None
    if lr_decay is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay)

    writer: SummaryWriter | None = None
    if tb_logdir:
        writer = SummaryWriter(log_dir=tb_logdir)

    range_epochs = range(n_epochs)
    for epoch in tqdm(range_epochs) if progress_bar else range_epochs:
        model.train()
        optimizer.zero_grad()

        # xy_train: features plus partially known targets for background nodes
        xy_train = torch.cat(
            [
                (
                    flatten(embeddings(x_tensor))
                    if periodic_embed_dim is not None
                    else x_tensor
                ),
                y_tensor.masked_fill(~backgnd_mask, 0).unsqueeze(1),
            ],
            dim=1,
        )

        preds = model(xy_train, r_tensor)
        loss = loss_fn(preds[probe_mask], y_tensor[probe_mask])
        loss.backward()
        optimizer.step()

        if writer:
            writer.add_scalar("train/loss", loss.item(), epoch)
            if scheduler:
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        if scheduler is not None:
            scheduler.step()

        if print_loss and epoch % 20 == 0:
            print(f"Epoch {epoch} - loss {loss.item():.4f}")
            print([layer.r_scale for layer in model.attn_layers])

    if writer:
        writer.close()
    
    model.eval()
    with torch.no_grad():
        preds = model(xy_train, r_tensor)
        y_val_true = y_tensor[val_mask].cpu().numpy()
        y_val_pred = preds[val_mask].cpu().numpy()

    mse = mean_squared_error(y_val_true, y_val_pred)
    r2 = r2_score(y_val_true, y_val_pred)
    mae = mean_absolute_error(y_val_true, y_val_pred)

    
    return mse, r2, mae, y_val_pred, y_val_true
    
