import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

def to_tensor(x_: npt.NDArray[np.float32]) -> torch.Tensor:
         return torch.tensor(x_, dtype=torch.float32)

def to_df(metrics_results: dict, decimal_places: int = 5) -> pd.DataFrame:
    results_np = {k: np.array([tup[:2] for tup in v]) for k, v in metrics_results.items()}
    results_means = [{"label": k, "means": v.mean(axis=0).round(decimals=decimal_places),"std": v.std(axis=0).round(decimal_places)} for k, v in results_np.items()]
    return pd.DataFrame(results_means)

