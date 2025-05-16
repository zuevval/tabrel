import torch


def mirror_triu(r: torch.Tensor) -> torch.Tensor:
    """
    Make a square matrix symmetric
    by mirroring its upper triangle to the lower triangle.

    Args:
        r (torch.Tensor): A 2D square tensor of shape (n, n).

    Returns:
        torch.Tensor: A symmetric tensor

    Raises:
        ValueError: If the input tensor is not square or not 2D.

    Example:
        >>> r = torch.tensor([[1, 2], [3, 4]])
        >>> mirror_triu(r)
        tensor([[1, 2],
                [2, 4]])
    """
    if len(r.shape) != 2 or r.shape[0] != r.shape[1]:
        raise ValueError(f"expecting a square 2D tensor, but r.shape={r.shape}")
    return torch.triu(r) + torch.triu(r, 1).T


def is_symmetric(r: torch.Tensor) -> bool:
    return (r == r.T).all().item()  # type:ignore
