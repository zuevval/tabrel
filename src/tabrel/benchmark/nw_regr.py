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
    clusters_backgnd: torch.Tensor
    x_query: torch.Tensor
    y_query_true: torch.Tensor
    clusters_query: torch.Tensor
    use_rel: bool

    def evaluate(self) -> dict[str, float]:
        if self.use_rel:
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
