from sklearn.datasets import fetch_california_housing

from tabrel.benchmark.nw_regr import Mlp, MlpConfig
from tabrel.utils.misc import to_tensor


def test_mlp() -> None:
    housing = fetch_california_housing()
    x = to_tensor(housing.data)
    out_dim = 8
    mlp = Mlp(MlpConfig(in_dim=x.shape[1], hidden_dim=64, out_dim=out_dim, dropout=0.1))
    assert mlp(x).shape == (len(x), out_dim)
