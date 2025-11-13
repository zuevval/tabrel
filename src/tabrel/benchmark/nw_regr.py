import logging
from dataclasses import dataclass
from itertools import product
from typing import Final

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tabrel.utils.misc import to_tensor

@dataclass(frozen=True)
class MlpConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int
    dropout: float

class Mlp(nn.Module):
    def __init__(self, 
                 config: MlpConfig,
                 ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.in_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class NwModelConfig:
    input_dim: int  # !when `mlp_config` is not None, this is the MLP out dim
    init_sigma: float = 0.1
    init_r_scale: float = 3.0
    trainable_weights_matrix: bool = False
    mlp_config: MlpConfig | None = None


class RelNwRegr(nn.Module):
    """
    Relationship-aware Nadaraya-Watson kernel regression
    """

    w: torch.Tensor | None
    x_transform: Mlp | None

    def __init__(self, cfg: NwModelConfig) -> None:
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor([float(cfg.init_sigma)]))
        self.r_scale = nn.Parameter(torch.tensor([float(cfg.init_r_scale)]))
        if cfg.trainable_weights_matrix:
            self.w = nn.Parameter(torch.ones((cfg.input_dim,)))
        else:
            self.w = None
        
        if cfg.mlp_config:
            self.x_transform = Mlp(cfg.mlp_config)
        else:
            self.x_transform = None

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
        if self.x_transform is not None:
            x_backgnd, x_query = self.x_transform(x_backgnd), self.x_transform(x_query)

        n_query, n_backgnd = r.shape
        # Expand x_backgnd and x_query to (n_query, n_backgnd, n_features)
        x_query_exp = x_query.unsqueeze(1).expand(
            -1, n_backgnd, -1
        )  # (n_query, n_backgnd, n_features)
        x_backgnd_exp = x_backgnd.unsqueeze(0).expand(
            n_query, -1, -1
        )  # (n_query, n_backgnd, n_features)

        if self.w is not None:
            x_dif = x_query_exp - x_backgnd_exp
            dists = torch.norm(x_dif * self.w, dim=2)

            # w_mtx = torch.eye(len(self.w)) * self.w**2
            # dists = torch.sqrt(batched_quadratic_form(x_dif, w_mtx))
        else:
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
    n_samples: int,
    n_clusters: int,
    seed: int,
    distr: str = "uniform",
    y_func: str = "square",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    if distr == "uniform":
        x = torch.rand((n_samples, 1)) * 2 - 1
    elif distr == "norm":
        x = torch.randn((n_samples, 1))
    else:
        raise ValueError(f"unknown distr type: {distr}")
    clusters = torch.randint(0, n_clusters, (n_samples,))
    if y_func == "square":
        y = x.flatten() ** 2
    elif y_func == "sign":
        y = torch.sign(x.flatten())
    elif y_func == "const":
        y = torch.zeros(len(x))
    else:
        raise ValueError(f"unknown `y_func`: {y_func}")
    y += clusters.float() * 0.5

    return x, y, clusters

def generate_multidim_noisy_data(n_samples: int, n_clusters: int, x_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, (n_samples, x_dim))
    clusters = np.random.randint(0, n_clusters, (n_samples,))
    y = np.sin(x[:, 0]) + np.cos(x[:, 1]) + x[:, 2] + clusters
    return x, y, clusters


def make_r(clusters: np.ndarray) -> np.ndarray:
    n = len(clusters)
    r = np.zeros((n, n))

    for i, j in product(range(n), range(n)):
        if clusters[i] == clusters[j] and np.random.choice((True, False)):
            r[i, j] = 1
    return r

def make_random_r(seed: int, clusters: np.ndarray) -> np.ndarray:
    np.random.seed(seed)
    n_samples = len(clusters)
    r = np.eye(n_samples)
    for i in range(n_samples):
        for j in range(i):
            if clusters[i] == clusters[j] and np.random.choice((True, False)):
                r[i, j] = 1
                r[j, i] = 1
    return r

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
    x_distr: str = "uniform"
    y_func: str = "square"


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

    y_val_true: np.ndarray | None = None
    y_val_pred: np.ndarray | None = None

    def evaluate(self) -> dict[str, float]:
        if (
            self.use_rel
            and self.clusters_backgnd is not None
            and self.clusters_query is not None
        ):
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
        distr=train_cfg.x_distr,
        y_func=train_cfg.y_func,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
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
    y_backgnd: npt.NDArray[np.float32],
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
    x_backgnd_norm = to_tensor(x_backgnd_norm)
    y_backgnd_torch = to_tensor(y_backgnd)
    x_query_norm = to_tensor(x_query_norm)
    y_query_torch = to_tensor(y_query)
    r_query_backgnd_torch = to_tensor(r_query_backgnd)
    x_val_norm = to_tensor(x_val_norm)
    x_nonval_norm = to_tensor(x_nonval_norm)
    y_nonval = to_tensor(y_nonval)
    r_val_nonval_torch = to_tensor(r_val_nonval)

    model = RelNwRegr(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(
            x_backgnd_norm,
            y_backgnd_torch,
            x_query_norm,
            r_query_backgnd_torch,
        )
        # TODO why errors here in run_training?
        loss = loss_fn(y_pred, y_query_torch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_val = model(
            x_nonval_norm,
            y_nonval,
            x_val_norm,
            r_val_nonval_torch,
        )
        y_pred_val_np = y_pred_val.numpy()

        mse = mean_squared_error(y_val, y_pred_val_np)
        r2 = r2_score(y_val, y_pred_val_np)
        mae = mean_absolute_error(y_val, y_pred_val_np)

        fitted_model = FittedNwRegr(
            model=model,
            x_backgnd=x_backgnd_norm,
            x_query=x_query_norm,
            y_backgnd=y_backgnd_torch,
            y_query_true=y_query_torch,
            clusters_query=None,
            clusters_backgnd=None,
            use_rel=None,
            y_val_true=y_val,
            y_val_pred=y_pred_val_np,
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
    mlp_config: MlpConfig | None = None,
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

    for use_rel, trainable_w, add_mlp in (
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (False, False, False),
        (False, False, True),
    ):
        if  add_mlp and mlp_config is None:
            continue
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
            cfg=NwModelConfig(
                input_dim=x.shape[1] if add_mlp or mlp_config is None else mlp_config.out_dim,
                trainable_weights_matrix=trainable_w,
                mlp_config=mlp_config,
            ),
            lr=lr,
            n_epochs=n_epochs,
        )
        results[f"rel={use_rel};trainable_w={trainable_w};mlp={add_mlp}"] = (
            mse,
            r2,
            mae,
            fitted_model,
        )

    # LightGBM
    x_train = x[train_indices]
    y_train = y[train_indices]
    lgb_params = {"objective": "regression", "metric": "rmse", "verbosity": -1}
    train_data = lgb.Dataset(x_train, label=y_train)
    model = lgb.train(lgb_params, train_data)
    y_pred = np.array(model.predict(x_val))
    results["lgb"] = (
        mean_squared_error(y_val, y_pred),
        r2_score(y_val, y_pred),
        mean_absolute_error(y_val, y_pred),
        None,
    )

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
            cfg=NwModelConfig(input_dim=x_broad.shape[1]),
            lr=lr,
            n_epochs=n_epochs,
        )
        results["rel-as-feats"] = (mse, r2, mae, fitted_model)

        x_train_broad = x_broad[train_indices]
        categorical_cols = list(range(x.shape[1], x.shape[1] + rel_as_feats.shape[1]))
        train_data_broad = lgb.Dataset(
            x_train_broad, label=y_train, categorical_feature=categorical_cols
        )
        model = lgb.train(lgb_params, train_data_broad)

        y_pred = np.array(model.predict(x_val_broad))
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        results["lgb-rel"] = (mse, r2, mae, None)

    return results


def metrics_mean(
    metrics: dict[str, list[np.ndarray]], lbls: list[str]
) -> dict[str, list[tuple[str, float]]]:
    return {
        k: list(zip(lbls, [round(arr.mean(), 3) for arr in v]))
        for k, v in metrics.items()
    }
