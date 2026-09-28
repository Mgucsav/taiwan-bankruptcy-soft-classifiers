"""Invariant tests for the Python ports of the public MATLAB classifiers."""

from __future__ import annotations

import numpy as np
import pytest

from taiwan_soft_classifiers.soft_classifiers import (
    FPFSKNNClassifier,
    IFPIFSHCClassifier,
    PFSKNNClassifier,
    _absolute_pearson_weights,
)


@pytest.fixture
def toy_binary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8], [1.0, 1.0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x_test = np.array([[0.05, 0.10], [0.95, 0.90]])
    return x_train, y_train, x_test


def test_absolute_pearson_weights_handles_constant_column() -> None:
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([0, 0, 1, 1])
    weights = _absolute_pearson_weights(x, y)
    assert weights[0] == 0.0
    assert 0.0 < weights[1] <= 1.0


@pytest.mark.parametrize(
    "model",
    [
        FPFSKNNClassifier(k=1, chunk_size=1),
        IFPIFSHCClassifier(chunk_size=1),
        PFSKNNClassifier(k=1, chunk_size=1),
    ],
)
def test_soft_classifier_separates_toy_groups(model: object, toy_binary: tuple) -> None:
    x_train, y_train, x_test = toy_binary
    prediction = model.fit(x_train, y_train).predict(x_test)
    np.testing.assert_array_equal(prediction, np.array([0, 1]))


def test_predictions_do_not_depend_on_test_batch_composition(toy_binary: tuple) -> None:
    x_train, y_train, x_test = toy_binary
    for model in [FPFSKNNClassifier(k=1), IFPIFSHCClassifier(), PFSKNNClassifier(k=1)]:
        model.fit(x_train, y_train)
        together = model.predict(x_test)
        separately = np.concatenate([model.predict(row.reshape(1, -1)) for row in x_test])
        np.testing.assert_array_equal(together, separately)
