import logging

import numpy as np


class RelDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    r_train: np.ndarray
    r_test_intra: np.ndarray
    r_test_inter: np.ndarray

    def __post_init__(
        self, x: np.ndarray, y: np.ndarray, r: np.ndarray, n_train: int
    ) -> None:
        (
            self.x_train,
            self.y_train,
        ) = (
            x[:n_train],
            y[:n_train],
        )
        self.x_test, self.y_test = x[n_train:], y[n_train:]
        self.r_train = r[:n_train, :n_train]
        self.r_test_intra = r[n_train:, n_train:]
        self.r_test_inter = r[:n_train, n_train:]

        logging.info(f"x shape: {x.shape}, y shape: {y.shape}, r shape: {r.shape}")
        logging.info(
            f"r_test_intra shape: {self.r_test_intra.shape}, "
            f"r_test_inter shape: {self.r_test_inter.shape}"
        )
