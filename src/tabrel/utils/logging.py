import logging
from datetime import datetime

from .config import TrainingConfig


def init_logging(config: TrainingConfig, level: int, print_logs: bool) -> None:
    handlers: list[logging.Handler] = [
        logging.FileHandler(config.log_dir / f"{datetime.now()}.log")
    ]

    if print_logs:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(level=level, handlers=handlers)
