from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


@dataclass(frozen=True)
class QueryUniqueBatchDataset(IterableDataset):
    x: torch.Tensor
    y: torch.Tensor
    query_size: int
    batch_size: int
    n_batches: int
    random_state: int

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(
                f"Dimension mismatch: len(x)={len(self.x)} != len(y)={len(self.y)}"
            )

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

            r = torch.eye(len(xb) + len(xq))  # relationship matrix
            yield xb, yb, xq, yq, r
