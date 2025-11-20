from dataclasses import dataclass

import numpy as np
import optuna
import torch

from tabrel.train import train_relnet


@dataclass(frozen=True)
class RelTrainData:
    r: np.ndarray
    x: np.ndarray
    y: np.ndarray
    back_ids: np.ndarray
    query_ids: np.ndarray
    val_ids: np.ndarray


def build_objective_relnet(
    trial: optuna.Trial,
    data: RelTrainData,
    n_epochs: int,
    seed: int,
) -> float:
    num_heads = trial.suggest_int("num_heads", 1, 20)
    embed_dim_factor = trial.suggest_int("embed_dim_factor", 1, 20)

    torch.manual_seed(seed)
    _, r2, _, _, _ = train_relnet(
        x=data.x,
        y=data.y,
        r=data.r,
        backgnd_indices=data.back_ids,
        query_indices=data.query_ids,
        val_indices=data.val_ids,
        n_epochs=n_epochs,
        lr=trial.suggest_float("lr", 1e-4, 1e-2),
        n_layers=trial.suggest_int("n_layers", 1, 3),
        embed_dim=num_heads * embed_dim_factor,
        periodic_embed_dim=trial.suggest_int("periodic_embed_dim", 1, 100),
        num_heads=num_heads,
        dropout=trial.suggest_float("dropout", 0, 0.5),
    )
    return r2
