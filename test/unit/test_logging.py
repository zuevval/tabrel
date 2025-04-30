import logging
import tempfile
from pathlib import Path

from tabrel.utils.config import TrainingConfig
from tabrel.utils.logging import init_logging


def test_init_logging() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir)
        config = TrainingConfig(
            batch_size=0,
            lr=0.0,
            n_epochs=0,
            log_dir=out_dir / "logs",
            checkpoints_dir=out_dir / "checkpoints",
            allow_dirs_exist=False,
        )
        init_logging(config, level=logging.INFO, print_logs=True)
        assert len(list(config.log_dir.glob("*.log"))) == 1
