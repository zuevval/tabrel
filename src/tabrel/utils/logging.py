import logging
from datetime import datetime

from tabrel.utils.config import TrainingConfig


def init_logging(config: TrainingConfig) -> None:
    handlers: list[logging.Handler] = [
        logging.FileHandler(config.log_dir / f"{datetime.now()}.log")
    ]

    if config.print_logs_to_console:
        handlers.append(logging.StreamHandler())

    logger = logging.getLogger()
    logger.setLevel(config.log_level)
    logger.handlers.clear()
    for h in handlers:
        logger.addHandler(h)
