"""Leakage-controlled Python ports of selected Samet Memiş MATLAB classifiers.

The algebra and default hyperparameters follow the authors' public MATLAB files. The one
intentional protocol correction is min-max fitting: the public demos normalise concatenated
training and test data, while these estimators fit ranges on training data only and clip held-out
values to [0, 1]. This prevents test-fold information from entering model fitting and preserves
the membership-value domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional feature matrix, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Feature matrix contains non-finite values")
    return array


def _as_1d_labels(y: np.ndarray, expected_rows: int) -> np.ndarray:
    labels = np.asarray(y).reshape(-1)
    if labels.shape[0] != expected_rows:
        raise ValueError(f"Received {labels.shape[0]} labels for {expected_rows} rows")
    return labels


def _absolute_pearson_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlations, matching MATLAB corr(...,'Rows','complete')."""

    x_centered = x - x.mean(axis=0, keepdims=True)
    y_float = y.astype(float)
    y_centered = y_float - y_float.mean()
    numerator = x_centered.T @ y_centered
    denominator = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )
    return np.abs(correlations)


@dataclass
class _TrainingMinMax:
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> _TrainingMinMax:
        return cls(minimum=x.min(axis=0), maximum=x.max(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        span = self.maximum - self.minimum
        nonconstant = span != 0
        result = np.ones_like(x, dtype=float)
        result[:, nonconstant] = (x[:, nonconstant] - self.minimum[nonconstant]) / span[nonconstant]
        return np.clip(result, 0.0, 1.0)


class _BaseSoftClassifier:
    """Minimal estimator interface shared by the source-faithful ports."""

    source_name: str

    def __init__(self, *, chunk_size: int = 128) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)

    def _fit_common(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_array = _as_2d_float(x)
        y_array = _as_1d_labels(y, x_array.shape[0])
        classes = np.unique(y_array)
        if classes.size < 2:
            raise ValueError("At least two classes are required")
        self.classes_ = classes
        self.y_train_ = y_array
        self.scaler_ = _TrainingMinMax.fit(x_array)
        self.x_train_scaled_ = self.scaler_.transform(x_array)
        self.n_features_in_ = x_array.shape[1]
        return x_array, y_array

    def _transform_test(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "scaler_"):
            raise RuntimeError("The classifier must be fitted before prediction")
        x_array = _as_2d_float(x)
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Received {x_array.shape[1]} features, expected {self.n_features_in_}"
            )
        return self.scaler_.transform(x_array)


class FPFSKNNClassifier(_BaseSoftClassifier):
    """FPFS-kNN port using the five metrics in the official ``FPFSkNN.m`` file."""

    source_name = "FPFS-kNN"

    def __init__(self, k: int = 3, *, chunk_size: int = 128) -> None:
        super().__init__(chunk_size=chunk_size)
        if k < 1:
            raise ValueError("k must be positive")
        self.k = int(k)

    def fit(self, x: np.ndarray, y: np.ndarray) -> FPFSKNNClassifier:
        x_raw, y_array = self._fit_common(x, y)
        self.feature_weights_ = _absolute_pearson_weights(x_raw, y_array)
        self.weighted_train_ = self.x_train_scaled_ * self.feature_weights_
        self.class_train_ = {
            label: self.weighted_train_[self.y_train_ == label] for label in self.classes_
        }
        return self

    @staticmethod
    def _k_mean(distance: np.ndarray, k: int) -> np.ndarray:
        effective_k = min(k, distance.shape[1])
        nearest = np.partition(distance, effective_k - 1, axis=1)[:, :effective_k]
        return nearest.mean(axis=1)

    def _class_metric_distances(self, weighted_test: np.ndarray) -> np.ndarray:
        # MATLAB metrics: Hamming, Chebyshev, Euclidean, Hamming-Hausdorff,
        # Minkowski(p=3). With the two-row fpfs matrices used by FPFSkNN.m,
        # Hamming-Hausdorff is algebraically identical to Chebyshev; both votes
        # are retained to reproduce the source algorithm's five-vote rule.
        result = np.empty((weighted_test.shape[0], self.classes_.size, 5), dtype=float)
        for class_index, label in enumerate(self.classes_):
            train = self.class_train_[label]
            hamming = cdist(weighted_test, train, metric="cityblock")
            chebyshev = cdist(weighted_test, train, metric="chebyshev")
            euclidean = cdist(weighted_test, train, metric="euclidean")
            minkowski3 = cdist(weighted_test, train, metric="minkowski", p=3)
            result[:, class_index, 0] = self._k_mean(hamming, self.k)
            result[:, class_index, 1] = self._k_mean(chebyshev, self.k)
            result[:, class_index, 2] = self._k_mean(euclidean, self.k)
            result[:, class_index, 3] = result[:, class_index, 1]
            result[:, class_index, 4] = self._k_mean(minkowski3, self.k)
        return result

    def predict(self, x: np.ndarray) -> np.ndarray:
        scaled = self._transform_test(x)
        predictions: list[np.ndarray] = []
        for start in range(0, scaled.shape[0], self.chunk_size):
            weighted = scaled[start : start + self.chunk_size] * self.feature_weights_
            distances = self._class_metric_distances(weighted)
            metric_winners = np.argmin(distances, axis=1)
            winner_indices = np.apply_along_axis(
                lambda row: np.bincount(row, minlength=self.classes_.size).argmax(),
                1,
                metric_winners,
            )
            predictions.append(self.classes_[winner_indices])
        return np.concatenate(predictions)


class IFPIFSHCClassifier(_BaseSoftClassifier):
    """IFPIFS-HC port following the official ``IFPIFSHC.m`` implementation."""

    source_name = "IFPIFS-HC"

    def __init__(
        self, lambda1: float = 5.0, lambda2: float = 0.5, *, chunk_size: int = 128
    ) -> None:
        super().__init__(chunk_size=chunk_size)
        if lambda1 <= 0 or lambda2 <= 0:
            raise ValueError("lambda1 and lambda2 must be positive")
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)

    def _embed(self, scaled: np.ndarray) -> np.ndarray:
        mu_x = 1.0 - (1.0 - scaled) ** self.lambda2
        nu_x = (1.0 - scaled) ** (self.lambda2 * (self.lambda2 + 1.0))
        pi_x = 1.0 - mu_x - nu_x
        return np.concatenate(
            [
                mu_x * self.mu_weight_,
                nu_x * self.nu_weight_,
                pi_x * self.pi_weight_,
            ],
            axis=1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> IFPIFSHCClassifier:
        x_raw, y_array = self._fit_common(x, y)
        correlation = _absolute_pearson_weights(x_raw, y_array)
        self.mu_weight_ = 1.0 - (1.0 - correlation) ** self.lambda1
        self.nu_weight_ = (1.0 - correlation) ** (self.lambda1 * (self.lambda1 + 1.0))
        self.pi_weight_ = 1.0 - self.mu_weight_ - self.nu_weight_
        self.embedded_train_ = self._embed(self.x_train_scaled_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        embedded = self._embed(self._transform_test(x))
        predictions: list[np.ndarray] = []
        for start in range(0, embedded.shape[0], self.chunk_size):
            distance = cdist(
                embedded[start : start + self.chunk_size],
                self.embedded_train_,
                metric="cityblock",
            )
            nearest = distance.argmin(axis=1)
            predictions.append(self.y_train_[nearest])
        return np.concatenate(predictions)


class PFSKNNClassifier(_BaseSoftClassifier):
    """PFS-kNN port following the official ``PFSkNN.m`` Minkowski formulation."""

    source_name = "PFS-kNN"

    def __init__(
        self,
        k: int = 3,
        lambda_value: float = 0.5,
        p: float = 5.0,
        *,
        chunk_size: int = 128,
    ) -> None:
        super().__init__(chunk_size=chunk_size)
        if k < 1 or lambda_value <= 0 or p <= 0:
            raise ValueError("k, lambda_value and p must be positive")
        self.k = int(k)
        self.lambda_value = float(lambda_value)
        self.p = float(p)

    def _embed(self, scaled: np.ndarray) -> np.ndarray:
        membership = 1.0 - (1.0 - scaled) ** self.lambda_value
        neutrality = scaled / self.lambda_value
        nonmembership = (1.0 - scaled) ** (self.lambda_value * (self.lambda_value + 1.0))
        refusal = 1.0 - membership - nonmembership
        return np.concatenate([membership, neutrality, nonmembership, refusal], axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> PFSKNNClassifier:
        self._fit_common(x, y)
        self.embedded_train_ = self._embed(self.x_train_scaled_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        embedded = self._embed(self._transform_test(x))
        predictions: list[np.ndarray] = []
        effective_k = min(self.k, self.embedded_train_.shape[0])
        encoded_train = np.searchsorted(self.classes_, self.y_train_)
        for start in range(0, embedded.shape[0], self.chunk_size):
            distance = cdist(
                embedded[start : start + self.chunk_size],
                self.embedded_train_,
                metric="minkowski",
                p=self.p,
            )
            nearest = np.argpartition(distance, effective_k - 1, axis=1)[:, :effective_k]
            encoded_neighbors = encoded_train[nearest]
            winner_indices = np.apply_along_axis(
                lambda row: np.bincount(row, minlength=self.classes_.size).argmax(),
                1,
                encoded_neighbors,
            )
            predictions.append(self.classes_[winner_indices])
        return np.concatenate(predictions)
