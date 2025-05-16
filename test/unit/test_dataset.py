from typing import Final

import pytest
import torch

from tabrel.dataset import QueryUniqueBatchDataset
from tabrel.train import generate_synthetic_data
from tabrel.utils.linalg import is_symmetric


def test_batch_shapes_and_uniqueness() -> None:
    n_features: Final[int] = 30
    x, y, r = generate_synthetic_data(
        num_samples=800, num_features=n_features, num_classes=2
    )
    batch_size = 20
    query_size = 10
    n_batches = 15
    ds = QueryUniqueBatchDataset(
        x, y, r, query_size, batch_size, n_batches, random_state=42
    )

    all_queries = []
    for xb, yb, xq, yq, r in ds:
        # Check shapes
        assert xb.shape == (batch_size, n_features)
        assert xq.shape == (query_size, n_features)
        assert yb.shape == (batch_size,)
        assert yq.shape == (query_size,)

        sample_size = batch_size + query_size
        assert r.shape == (sample_size, sample_size)
        assert is_symmetric(r)

        # Accumulate queries
        all_queries.append(xq)

    # Check global uniqueness of query samples
    all_queries_tensor = torch.cat(all_queries, dim=0)
    as_tuples = [tuple(row.tolist()) for row in all_queries_tensor]
    assert len(as_tuples) == len(
        set(as_tuples)
    ), "Query samples are not unique across batches"

    # Check number of batches
    assert len(all_queries) == n_batches


def test_valueerror_r_y_problems() -> None:
    sample_size: Final[int] = 100
    x = torch.randn(sample_size, 10)
    y = torch.randint(low=0, high=2, size=(sample_size,))
    r = torch.eye(sample_size)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        _ = QueryUniqueBatchDataset(
            x, y[:-1], r, query_size=5, batch_size=5, n_batches=1, random_state=0
        )

    with pytest.raises(ValueError, match="wrong r.shape"):
        _ = QueryUniqueBatchDataset(
            x, y, r[:, :-1], query_size=5, batch_size=5, n_batches=1, random_state=0
        )

    r[0, 1] = 1  # make r asymmetric
    with pytest.raises(ValueError, match="r must be symmetric"):
        _ = QueryUniqueBatchDataset(
            x, y, r, query_size=5, batch_size=5, n_batches=1, random_state=0
        )


def test_valueerror_insufficient_data() -> None:
    sample_size: Final[int] = 50
    x = torch.randn(sample_size, 10)
    y = torch.randint(low=0, high=2, size=(sample_size,))
    r = torch.eye(sample_size)
    with pytest.raises(ValueError, match="Not enough data"):
        _ = QueryUniqueBatchDataset(
            x, y, r, query_size=10, batch_size=10, n_batches=5, random_state=0
        )


def test_determinism() -> None:
    x, y, r = generate_synthetic_data(num_samples=500, num_features=10, num_classes=3)
    ds1 = QueryUniqueBatchDataset(
        x, y, r, query_size=5, batch_size=5, n_batches=3, random_state=123
    )
    ds2 = QueryUniqueBatchDataset(
        x, y, r, query_size=5, batch_size=5, n_batches=3, random_state=123
    )

    for (xb1, yb1, xq1, yq1, r1), (xb2, yb2, xq2, yq2, r2) in zip(ds1, ds2):
        assert torch.equal(xb1, xb2)
        assert torch.equal(xq1, xq2)
        assert torch.equal(yb1, yb2)
        assert torch.equal(yq1, yq2)
        assert torch.equal(r1, r2)
