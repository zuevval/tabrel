import logging
from dataclasses import dataclass
from typing import Final

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class NwModelConfig:
    init_sigma: float = 0.1
    init_r_scale: float = 3.0


class RelNwRegr(nn.Module):
    """
    Relationship-aware Nadaraya-Watson kernel regression
    """

    def __init__(self, cfg: NwModelConfig) -> None:
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor([float(cfg.init_sigma)]))
        self.r_scale = nn.Parameter(torch.tensor([float(cfg.init_r_scale)]))

    def forward(
        self,
        x_backgnd: torch.Tensor,  # (n_backgnd, n_features)
        y_backgnd: torch.Tensor,  # (n_backgnd,)
        x_query: torch.Tensor,  # (n_query, n_features)
        r: torch.Tensor,  # (n_query, n_backgnd)
    ) -> torch.Tensor:
        """
        Returns predicted y: (n_query,)
        """
        n_query, n_backgnd = r.shape
        # Expand x_backgnd and x_query to (n_query, n_backgnd, n_features)
        x_query_exp = x_query.unsqueeze(1).expand(
            -1, n_backgnd, -1
        )  # (n_query, n_backgnd, n_features)
        x_backgnd_exp = x_backgnd.unsqueeze(0).expand(
            n_query, -1, -1
        )  # (n_query, n_backgnd, n_features)

        # Compute L2 distances: (n_query, n_backgnd)
        dists = torch.norm(x_query_exp - x_backgnd_exp, dim=2)

        # Compute kernel weights: (n_query, n_backgnd)
        k_vals = torch.exp(-dists / self.sigma + self.r_scale * r)

        # Normalize weights
        k_sum = k_vals.sum(dim=1, keepdim=True) + 1e-8  # avoid division by zero
        weights = k_vals / k_sum  # (n_query, n_backgnd)

        # Weighted sum of y_backgnd: (n_query,)
        y_pred = torch.matmul(weights, y_backgnd)

        return y_pred


def generate_toy_regr_data(
    n_samples: int, n_clusters: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.rand((n_samples, 1)) * 2 - 1
    clusters = torch.randint(0, n_clusters, (n_samples,))
    y = x.flatten() ** 2 + clusters.float() * 0.5

    return x, y, clusters


def compute_relation_matrix(
    backgnd_clusters: torch.Tensor,
    query_clusters: torch.Tensor,  # (n_backgnd,)  # (n_query,)
) -> torch.Tensor:
    """
    Computes binary relation matrix indicating cluster match.

    Returns:
        r: (n_query, n_backgnd)
    """
    return (query_clusters.unsqueeze(1) == backgnd_clusters.unsqueeze(0)).float()


@dataclass(frozen=True)
class NwTrainConfig:
    n_backgnd: int = 50
    n_query: int = 30
    n_clusters: int = 3
    seed: int = 42
    n_epochs: int = 40
    lr: float = 1e-2
    use_rel: bool = True


@dataclass(frozen=True)
class FittedNwRegr:
    model: RelNwRegr
    x_backgnd: torch.Tensor
    y_backgnd: torch.Tensor
    clusters_backgnd: torch.Tensor | None
    x_query: torch.Tensor
    y_query_true: torch.Tensor
    clusters_query: torch.Tensor | None
    use_rel: bool | None

    def evaluate(self) -> dict[str, float]:
        if self.use_rel and self.clusters_backgnd and self.clusters_query:
            r = compute_relation_matrix(self.clusters_backgnd, self.clusters_query)
        else:
            r = torch.zeros((len(self.x_query), len(self.x_backgnd)))

        with torch.no_grad():
            y_pred = self.model(self.x_backgnd, self.y_backgnd, self.x_query, r)

        y_true_np = self.y_query_true.detach().numpy()
        y_pred_np = y_pred.detach().numpy()

        return {
            "r2": r2_score(y_true_np, y_pred_np),
            "mae": mean_absolute_error(y_true_np, y_pred_np),
            "mse": mean_squared_error(y_true_np, y_pred_np),
        }


def train_nw(model_cfg: NwModelConfig, train_cfg: NwTrainConfig) -> FittedNwRegr:
    x, y, clusters = generate_toy_regr_data(
        n_samples=train_cfg.n_backgnd + train_cfg.n_query,
        n_clusters=train_cfg.n_clusters,
        seed=train_cfg.seed,
    )

    n_backgnd: Final[int] = train_cfg.n_backgnd
    x_backgnd, y_backgnd, clusters_backgnd = (
        x[:n_backgnd],
        y[:n_backgnd],
        clusters[:n_backgnd],
    )
    x_query, y_query_true, clusters_query = (
        x[n_backgnd:],
        y[n_backgnd:],
        clusters[n_backgnd:],
    )
    if train_cfg.use_rel:
        r = compute_relation_matrix(clusters_backgnd, clusters_query)
    else:
        r = torch.zeros((train_cfg.n_query, train_cfg.n_backgnd))

    model = RelNwRegr(cfg=model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    loss_fn = nn.MSELoss()

    torch.manual_seed(train_cfg.seed)
    model.train()
    for epoch in range(train_cfg.n_epochs):
        optimizer.zero_grad()
        y_pred = model(x_backgnd, y_backgnd, x_query, r)
        loss = loss_fn(torch.Tensor(y_pred), y_query_true)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logging.info(
                f"Epoch {epoch+1}/{train_cfg.n_epochs}, "
                f"Loss: {loss.item():.4f}, Sigma: {model.sigma.item():.4f}, "
                f"R_scale: {model.r_scale.item():.4f}"
            )

    model.eval()
    return FittedNwRegr(
        model=model,
        x_backgnd=x_backgnd,
        y_backgnd=y_backgnd,
        clusters_backgnd=clusters_backgnd,
        x_query=x_query,
        y_query_true=y_query_true,
        clusters_query=clusters_query,
        use_rel=train_cfg.use_rel,
    )


def train_nw_arbitrary(
    x_backgnd: np.ndarray,
    y_backgnd: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    r_query_backgnd: np.ndarray,
    r_val_nonval: np.ndarray,
    cfg: NwModelConfig,
    lr: float,
    n_epochs: int,
) -> tuple[float, float, float, FittedNwRegr]:
    x_mean = np.mean(x_backgnd, axis=0, keepdims=True)
    x_std = np.std(x_backgnd, axis=0, keepdims=True) + 1e-8

    x_backgnd_norm = (x_backgnd - x_mean) / x_std
    x_query_norm = (x_query - x_mean) / x_std
    x_val_norm = (x_val - x_mean) / x_std

    x_nonval_norm = np.concatenate((x_backgnd_norm, x_query_norm))
    y_nonval = np.concatenate((y_backgnd, y_query))

    # Convert to torch
    x_backgnd_norm = torch.tensor(x_backgnd_norm, dtype=torch.float32)
    y_backgnd = torch.tensor(y_backgnd, dtype=torch.float32)
    x_query_norm = torch.tensor(x_query_norm, dtype=torch.float32)
    y_query = torch.tensor(y_query, dtype=torch.float32)
    r_query_backgnd = torch.tensor(r_query_backgnd, dtype=torch.float32)
    x_val_norm = torch.tensor(x_val_norm, dtype=torch.float32)
    x_nonval_norm = torch.tensor(x_nonval_norm, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_nonval = torch.tensor(y_nonval, dtype=torch.float32)
    r_val_nonval = torch.tensor(r_val_nonval, dtype=torch.float32)

    model = RelNwRegr(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(
            x_backgnd_norm,
            y_backgnd,
            x_query_norm,
            r_query_backgnd,
        )
        loss = loss_fn(y_pred, y_query)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_val = model(
            x_nonval_norm,
            y_nonval,
            x_val_norm,
            r_val_nonval,
        )
        y_pred_val_np = y_pred_val.numpy()
        y_val_np = y_val.numpy()

        mse = mean_squared_error(y_val_np, y_pred_val_np)
        r2 = r2_score(y_val_np, y_pred_val_np)
        mae = mean_absolute_error(y_val_np, y_pred_val_np)

        fitted_model = FittedNwRegr(
            model=model,
            x_backgnd=x_backgnd_norm,
            x_query=x_query_norm,
            y_backgnd=y_backgnd,
            y_query_true=y_query,
            clusters_query=None,
            clusters_backgnd=None,
            use_rel=None,
        )

        return mse, r2, mae, fitted_model


def run_training(
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    backgnd_indices: np.ndarray,
    query_indices: np.ndarray,
    val_indices: np.ndarray,
    lr: float,
    n_epochs: int,
    rel_as_feats: np.ndarray | None = None,
) -> dict[str, tuple[float, float, float, FittedNwRegr | None]]:
    x_backgnd = x[backgnd_indices]
    y_backgnd = y[backgnd_indices]
    x_query = x[query_indices]
    y_query = y[query_indices]
    x_val = x[val_indices]
    y_val = y[val_indices]

    train_indices = np.concatenate((backgnd_indices, query_indices))

    r_query_backgnd = r[np.ix_(query_indices, backgnd_indices)]
    r_val_nonval = r[np.ix_(val_indices, train_indices)]

    results: dict[str, tuple[float, float, float, FittedNwRegr | None]] = {}
    model_cfg = NwModelConfig()

    for use_rel in (True, False):
        mse, r2, mae, fitted_model = train_nw_arbitrary(
            x_backgnd=x_backgnd,
            y_backgnd=y_backgnd,
            x_query=x_query,
            y_query=y_query,
            x_val=x_val,
            y_val=y_val,
            r_query_backgnd=(
                r_query_backgnd if use_rel else np.zeros_like(r_query_backgnd)
            ),
            r_val_nonval=r_val_nonval if use_rel else np.zeros_like(r_val_nonval),
            cfg=model_cfg,
            lr=lr,
            n_epochs=n_epochs,
        )
        results[f"rel={use_rel}"] = (mse, r2, mae, fitted_model)

    if rel_as_feats is not None:
        x_broad = np.concatenate((x, rel_as_feats), axis=1)
        x_backgnd_broad = x_broad[backgnd_indices]
        x_query_broad = x_broad[query_indices]
        x_val_broad = x_broad[val_indices]

        mse, r2, mae, fitted_model = train_nw_arbitrary(
            x_backgnd=x_backgnd_broad,
            y_backgnd=y_backgnd,
            x_query=x_query_broad,
            y_query=y_query,
            x_val=x_val_broad,
            y_val=y_val,
            r_query_backgnd=np.zeros_like(r_query_backgnd),
            r_val_nonval=np.zeros_like(r_val_nonval),
            cfg=model_cfg,
            lr=lr,
            n_epochs=n_epochs,
        )
        results["rel-as-feats"] = (mse, r2, mae, fitted_model)

        x_train_broad = x_broad[train_indices]
        y_train = y[train_indices]

        train_data = lgb.Dataset(x_train_broad, label=y_train)

        params = {"objective": "regression", "metric": "rmse", "verbosity": -1}
        model = lgb.train(params, train_data)

        y_pred = model.predict(x_val_broad)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        results["lgb"] = (mse, r2, mae, None)

    return results
