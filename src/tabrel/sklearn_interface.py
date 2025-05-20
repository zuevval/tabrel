import logging
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted  # type:ignore

from tabrel.model import TabularTransformerClassifierModel
from tabrel.train import run_epoch, wrap_data
from tabrel.utils.config import ProjectConfig
from tabrel.utils.linalg import is_symmetric, mirror_triu
from tabrel.utils.logging import init_logging


@dataclass(frozen=True)
class FitData:
    model: TabularTransformerClassifierModel
    x_train: torch.Tensor
    y_train: torch.Tensor
    r_train: torch.Tensor


class TabRelClassifier(ClassifierMixin, BaseEstimator):
    """
    A scikit-learn classifier wrapper for TabularTransformerClassifierModel
    """

    config: ProjectConfig
    device: torch.device

    def __init__(self, config: ProjectConfig) -> None:
        init_logging(config.training)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray, r: np.ndarray) -> None:
        """
        Fits the classifier to the training data

        Args:
            X (np.ndarray): The training data (two-dimensional array)
            y (np.ndarray): 1D array of integer training class labels
            r (np.ndarray): relationships between samples
        """
        self.classes_ = np.unique(y)  # may be used by scikit-learn later
        x_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.long)
        r_train = torch.tensor(r, dtype=torch.float32)
        train_data = wrap_data(
            x=x_train, y=y_train, r=r_train, config=self.config.training
        )
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=None)

        model = TabularTransformerClassifierModel(self.config.model).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.training.lr)

        for _ in range(self.config.training.n_epochs):
            run_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                device=self.device,
                optimizer=optimizer,
            )

        self.fit_data_ = FitData(
            model=model, x_train=x_train, y_train=y_train, r_train=r_train
        )

    def predict(
        self, X: np.ndarray, r_inter: np.ndarray, r_intra: np.ndarray
    ) -> np.ndarray:
        """
        Predicts the labels for the given data

        Args:
            X (np.ndarray): The data to predict on
            r_inter (np.ndarray): Relationships between elements of X
            r_intra (np.ndarray): Relationships between X and X_train

        Returns:
            np.ndarray: The predicted labels
        """
        check_is_fitted(self, attributes=["fit_data_"])

        x_query = torch.tensor(X, dtype=torch.float32)

        n_query_samples: Final[int] = len(x_query)
        n_train_samples: Final[int] = len(self.fit_data_.x_train)
        if r_inter.shape != (n_query_samples, n_query_samples):
            raise ValueError("`r_inter` must be a square matrix len(X) x len(X)")
        if r_intra.shape != (n_train_samples, n_query_samples):
            raise ValueError("`r_intra` must be of shape (len(X_train), len(X))")
        if not is_symmetric(r_inter):
            raise ValueError("`r_inter` must be symmetric")

        r = torch.eye(n_train_samples + n_query_samples)
        r[n_train_samples:, n_train_samples:] = torch.tensor(r_inter)
        r[:n_train_samples, n_train_samples:] = torch.tensor(r_intra)
        r = mirror_triu(r)

        with torch.no_grad():
            outputs = self.fit_data_.model(
                xb=self.fit_data_.x_train, yb=self.fit_data_.y_train, xq=x_query, r=r
            )
            _, predicted = torch.max(outputs, 1)
            return predicted.numpy()  # type:ignore


class DummyTabRelClassifier(ClassifierMixin, BaseEstimator):
    """
    TabRelClassifier with dummy relationship matrices.
    Mainly for testing and benchmarking
    """

    _classifier: TabRelClassifier

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        # set rel=False regardless to passed settings (do not use relationships)
        if config.model.rel:
            logging.warning("config.model.rel=True will be ignored")
        config = replace(config, model=replace(config.model, rel=False))
        self._classifier = TabRelClassifier(config)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # any `r` is OK - it is not used anyway
        self._classifier.fit(X, y=y, r=np.eye(len(X)))
        self.classes_ = np.unique(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_train, n_test = len(self._classifier.fit_data_.x_train), len(X)
        return self._classifier.predict(
            X, r_inter=np.eye(n_test), r_intra=np.zeros((n_train, n_test))
        )
