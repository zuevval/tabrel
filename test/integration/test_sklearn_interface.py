from pathlib import Path
from test.utils import light_config, make_test_dir
from typing import Final

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError  # type:ignore

from tabrel.sklearn_interface import TabularTransformerClassifier
from tabrel.utils.config import ProjectConfig


def test_classifier_fit_predict(request: pytest.FixtureRequest) -> None:
    """Test the fit and predict methods with valid data."""
    out_dir: Final[Path] = make_test_dir(request)
    num_features: Final[int] = 1
    num_classes: Final[int] = 2
    train_size: Final[int] = 300

    config = light_config(
        out_dir=out_dir, num_features=num_features, num_classes=num_classes
    )
    X_train = np.random.rand(train_size, num_features)
    y_train = np.random.randint(0, num_classes, train_size)
    X_test = np.random.rand(20, num_features)

    classifier = TabularTransformerClassifier(config)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    assert predictions.shape == (len(X_test),)
    assert np.all(predictions >= 0) and np.all(predictions < num_classes)


def test_classifier_predict_before_fit() -> None:
    X_test = np.random.rand(20, 10)
    classifier = TabularTransformerClassifier(ProjectConfig.default())

    with pytest.raises(NotFittedError):
        classifier.predict(X_test)
