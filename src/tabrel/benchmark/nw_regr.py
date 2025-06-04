import logging
from dataclasses import dataclass
from typing import Final

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
        x_train: torch.Tensor,  # (n_train, n_features)
        y_train: torch.Tensor,  # (n_train,)
        x_test: torch.Tensor,  # (n_test, n_features)
        r: torch.Tensor,  # (n_test, n_train)
    ) -> torch.Tensor:
        """
        Returns predicted y: (n_test,)
        """
        n_test, n_train = r.shape
        # Expand x_train and x_test to (n_test, n_train, n_features)
        x_test_exp = x_test.unsqueeze(1).expand(
            -1, n_train, -1
        )  # (n_test, n_train, n_features)
        x_train_exp = x_train.unsqueeze(0).expand(
            n_test, -1, -1
        )  # (n_test, n_train, n_features)

        # Compute L2 distances: (n_test, n_train)
        dists = torch.norm(x_test_exp - x_train_exp, dim=2)

        # Compute kernel weights: (n_test, n_train)
        k_vals = torch.exp(-dists / self.sigma + self.r_scale * r)

        # Normalize weights
        k_sum = k_vals.sum(dim=1, keepdim=True) + 1e-8  # avoid division by zero
        weights = k_vals / k_sum  # (n_test, n_train)

        # Weighted sum of y_train: (n_test,)
        y_pred = torch.matmul(weights, y_train)

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
    train_clusters: torch.Tensor, test_clusters: torch.Tensor  # (n_train,)  # (n_test,)
) -> torch.Tensor:
    """
    Computes binary relation matrix indicating cluster match.

    Returns:
        r: (n_test, n_train)
    """
    return (test_clusters.unsqueeze(1) == train_clusters.unsqueeze(0)).float()


@dataclass(frozen=True)
class NwTrainConfig:
    n_train: int = 50
    n_test: int = 30
    n_clusters: int = 3
    seed: int = 42
    n_epochs: int = 40
    lr: float = 1e-2
    use_rel: bool = True


@dataclass(frozen=True)
class FittedNwRegr:
    model: RelNwRegr
    x_train: torch.Tensor
    y_train: torch.Tensor
    clusters_train: torch.Tensor
    x_test: torch.Tensor
    y_test_true: torch.Tensor
    clusters_test: torch.Tensor
    use_rel: bool

    def evaluate(self) -> dict[str, float]:
        if self.use_rel:
            r = compute_relation_matrix(self.clusters_train, self.clusters_test)
        else:
            r = torch.zeros((len(self.x_test), len(self.x_train)))

        with torch.no_grad():
            y_pred = self.model(self.x_train, self.y_train, self.x_test, r)

        y_true_np = self.y_test_true.detach().numpy()
        y_pred_np = y_pred.detach().numpy()

        return {
            "r2": r2_score(y_true_np, y_pred_np),
            "mae": mean_absolute_error(y_true_np, y_pred_np),
            "mse": mean_squared_error(y_true_np, y_pred_np),
        }


def train_nw(model_cfg: NwModelConfig, train_cfg: NwTrainConfig) -> FittedNwRegr:
    x, y, clusters = generate_toy_regr_data(
        n_samples=train_cfg.n_train + train_cfg.n_test,
        n_clusters=train_cfg.n_clusters,
        seed=train_cfg.seed,
    )

    n_train: Final[int] = train_cfg.n_train
    x_train, y_train, clusters_train = x[:n_train], y[:n_train], clusters[:n_train]
    x_test, y_test_true, clusters_test = x[n_train:], y[n_train:], clusters[n_train:]
    if train_cfg.use_rel:
        r = compute_relation_matrix(clusters_train, clusters_test)
    else:
        r = torch.zeros((train_cfg.n_test, train_cfg.n_train))

    model = RelNwRegr(cfg=model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    loss_fn = nn.MSELoss()

    torch.manual_seed(train_cfg.seed)
    model.train()
    for epoch in range(train_cfg.n_epochs):
        optimizer.zero_grad()
        y_pred = model(x_train, y_train, x_test, r)
        loss = loss_fn(torch.Tensor(y_pred), y_test_true)
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
        x_train=x_train,
        y_train=y_train,
        clusters_train=clusters_train,
        x_test=x_test,
        y_test_true=y_test_true,
        clusters_test=clusters_test,
        use_rel=train_cfg.use_rel,
    )
