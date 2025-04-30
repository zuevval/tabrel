import logging
from datetime import datetime

from tabrel.utils.config import TrainingConfig


def init_logging(config: TrainingConfig) -> None:
    handlers: list[logging.Handler] = [
        logging.FileHandler(config.log_dir / f"{datetime.now()}.log")
    ]

    if config.print_logs_to_console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(level=config.log_level, handlers=handlers)
