from pathlib import Path
from test.utils import light_config, make_test_dir
from typing import Final

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError  # type:ignore

from tabrel.sklearn_interface import DummyTabRelClassifier, TabRelClassifier
from tabrel.utils.config import ProjectConfig


def make_sample(
    num_features: int, num_classes: int, train_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.random.rand(train_size, num_features)
    y_train = np.random.randint(0, num_classes, train_size)
    X_test = np.random.rand(20, num_features)
    return X_train, y_train, X_test


def test_classifier_fit_predict(request: pytest.FixtureRequest) -> None:
    """Test the fit and predict methods with valid data."""
    out_dir: Final[Path] = make_test_dir(request)
    num_features: Final[int] = 1
    num_classes: Final[int] = 2
    train_size: Final[int] = 300

    config = light_config(
        out_dir=out_dir, num_features=num_features, num_classes=num_classes
    )
    X_train, y_train, X_test = make_sample(
        num_features=num_features, num_classes=num_classes, train_size=train_size
    )

    n_train, n_test = len(X_train), len(X_test)
    r_train = np.eye(n_train)
    r_test_intra = np.eye(n_test)
    r_test_inter = np.zeros((n_train, n_test))

    classifier = TabRelClassifier(config)
    classifier.fit(X_train, y_train, r_train)
    predictions = classifier.predict(X_test, r_inter=r_test_inter, r_intra=r_test_intra)

    assert predictions.shape == (len(X_test),)
    assert np.all(predictions >= 0) and np.all(predictions < num_classes)

    # test problematic relationship matrices
    with pytest.raises(ValueError, match="`r_intra` must be a square matrix"):
        _ = classifier.predict(X_test, r_intra=r_test_intra[:-1], r_inter=r_test_inter)

    with pytest.raises(ValueError, match="`r_inter` must be of shape"):
        _ = classifier.predict(X_test, r_inter=r_test_inter[:-1], r_intra=r_test_intra)

    r_test_intra[0, -1] += 1  # make r asymmetric
    with pytest.raises(ValueError, match="`r_intra` must be symmetric"):
        _ = classifier.predict(X_test, r_inter=r_test_inter, r_intra=r_test_intra)


def test_classifier_predict_before_fit() -> None:
    n_samples: Final[int] = 20
    X_test = np.random.rand(n_samples, 10)
    classifier = TabRelClassifier(ProjectConfig.default())

    with pytest.raises(NotFittedError):
        classifier.predict(
            X_test, r_inter=np.zeros((5, n_samples)), r_intra=np.eye(n_samples)
        )


def test_dummy_tabrel_classifier(request: pytest.FixtureRequest) -> None:
    out_dir: Final[Path] = make_test_dir(request)
    num_features: Final[int] = 1
    num_classes: Final[int] = 2
    train_size: Final[int] = 200

    X_train, y_train, X_test = make_sample(
        num_features=num_features, num_classes=num_classes, train_size=train_size
    )
    config = light_config(
        out_dir=out_dir, num_features=num_features, num_classes=num_classes
    )
    classifier = DummyTabRelClassifier(config)

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    assert predictions.shape == (len(X_test),)
