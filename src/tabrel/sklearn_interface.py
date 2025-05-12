from dataclasses import dataclass

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted

from tabrel.model import TabularTransformerClassifierModel
from tabrel.train import run_epoch, wrap_data
from tabrel.utils.config import ProjectConfig
from tabrel.utils.logging import init_logging


@dataclass(frozen=True)
class FitData:
    model: TabularTransformerClassifierModel
    x_train: torch.Tensor
    y_train: torch.Tensor


class TabularTransformerClassifier(ClassifierMixin, BaseEstimator):
    """
    A scikit-learn classifier wrapper for TabularTransformerClassifierModel
    """

    config: ProjectConfig
    device: torch.device

    def __init__(self, config: ProjectConfig) -> None:
        init_logging(config.training)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the classifier to the training data

        Args:
            X (np.ndarray): The training data (two-dimensional array)
            y (np.ndarray): 1D array of integer training class labels
        """
        self.classes_ = np.unique(y)  # may be used by scikit-learn later
        x_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.long)
        train_data = wrap_data(x=x_train, y=y_train, config=self.config.training)
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

        self.fit_data_ = FitData(model=model, x_train=x_train, y_train=y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the given data

        Args:
            X (np.ndarray): The data to predict on

        Returns:
            np.ndarray: The predicted labels
        """
        x_query = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            check_is_fitted(self, attributes=["fit_data_"])
            outputs = self.fit_data_.model(
                xb=self.fit_data_.x_train, yb=self.fit_data_.y_train, xq=x_query
            )
            _, predicted = torch.max(outputs, 1)
            return predicted.numpy()
