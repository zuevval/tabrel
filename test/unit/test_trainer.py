from test.utils import light_config, make_test_dir

import pytest
import torch

from tabrel.train import load_checkpoint
from tabrel.utils.logging import init_logging


def test_load_checkpoint_not_found(request: pytest.FixtureRequest) -> None:
    out_dir = make_test_dir(request)
    config = light_config(out_dir, num_features=1, num_classes=2)
    init_logging(config.training)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert out_dir.exists()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(config, device)
