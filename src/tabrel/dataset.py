from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from tabrel.utils.linalg import is_symmetric


@dataclass(frozen=True)
class QueryUniqueBatchDataset(IterableDataset):
    x: torch.Tensor
    y: torch.Tensor
    r: torch.Tensor  # n_samples x n_samples relationships matrix
    query_size: int
    batch_size: int
    n_batches: int
    random_state: int

    def __post_init__(self) -> None:
        n_samples = len(self.x)
        if n_samples != len(self.y):
            raise ValueError(
                f"Dimension mismatch: len(x)={n_samples} != len(y)={len(self.y)}"
            )

        expected_r_shape = (n_samples, n_samples)
        if self.r.shape != expected_r_shape:
            raise ValueError(
                f"wrong r.shape: {self.r.shape}, should be {expected_r_shape}"
            )

        if not is_symmetric(self.r):
            raise ValueError("r must be symmetric")

        if len(self.x) < self.query_size * self.n_batches + self.batch_size:
            raise ValueError(
                f"Not enough data in x for batches: len(x)={len(self.x)}, "
                f"should be >= {self.batch_size} + {self.query_size} * {self.n_batches}"
            )

    def __iter__(
        self,
    ) -> Iterator[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        g = torch.Generator().manual_seed(self.random_state)

        perm = torch.randperm(len(self.x), generator=g)
        total_query_size = self.query_size * self.n_batches
        query_indices = perm[:total_query_size]
        remaining_indices = perm[total_query_size:]

        for i in range(self.n_batches):
            q_idx = query_indices[i * self.query_size : (i + 1) * self.query_size]
            xq = self.x[q_idx]
            yq = self.y[q_idx]

            # Random support set (can repeat across batches)
            support_perm = torch.randperm(len(remaining_indices), generator=g)[
                : self.batch_size
            ]
            b_idx = remaining_indices[support_perm]
            xb = self.x[b_idx]
            yb = self.y[b_idx]

            s_idx = torch.cat((q_idx, b_idx))
            r = self.r[s_idx][:, s_idx]  # relationship sub-matrix
            yield xb, yb, xq, yq, r
