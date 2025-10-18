import pytest
import torch

from tabrel.utils.linalg import batched_quadratic_form, is_symmetric, mirror_triu


def test_mirror_triu() -> None:
    # 2x2 matrix
    r = torch.tensor([[1, 2], [3, 4]])
    expected = torch.tensor([[1, 2], [2, 4]])
    assert torch.equal(mirror_triu(r), expected)

    # 3x3 matrix
    r = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected = torch.tensor([[1, 2, 3], [2, 5, 6], [3, 6, 9]])
    assert torch.equal(mirror_triu(r), expected)

    # Non-square matrix
    with pytest.raises(ValueError, match="expecting a square 2D tensor"):
        mirror_triu(torch.randn(3, 4))

    # 1D tensor
    with pytest.raises(ValueError, match="expecting a square 2D tensor"):
        mirror_triu(torch.tensor([1, 2, 3]))


def test_is_symmetric() -> None:
    r = torch.tensor([[1, 2], [2, 1]])
    assert is_symmetric(r)

    r = torch.tensor([[1, 2], [3, 4]])
    assert not is_symmetric(r)

    # Edge case - single-element tensor (always symmetric)
    r = torch.tensor([[5]])
    assert is_symmetric(r)


def test_batched_quadratic_norm() -> None:
    x = torch.randn(2, 3, 4)
    w = torch.randn(4, 4)
    assert batched_quadratic_form(x, w).shape == (2, 3)

    x, w = torch.Tensor([[[1, 1], [1, 1]]]), torch.Tensor([[1, 1], [1, 1]])
    result = batched_quadratic_form(x, w)
    assert result.shape == (1, 2)
    assert result[0][0] == result[0][1] == 4
