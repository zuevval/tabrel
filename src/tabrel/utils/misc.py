import numpy as np
import numpy.typing as npt
import torch

def to_tensor(x_: npt.NDArray[np.float32]) -> torch.Tensor:
         return torch.tensor(x_, dtype=torch.float32)
