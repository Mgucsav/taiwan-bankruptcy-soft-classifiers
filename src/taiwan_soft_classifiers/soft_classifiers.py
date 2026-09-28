"""Leakage-controlled Python implementations of three Samet Memiş soft classifiers.

- FPFS-kNN and IFPIFS-HC are leakage-controlled Python adaptations of the transformations and
  decision rules in the authors' public MATLAB files (FPFS-kNN @ 776f69c, IFPIFS-HC @ a9d3093).
- PFS-kNN is a paper-concordant Python implementation (Memiş, 2023, Electronics 12(19), 4129,
  Algorithm 1) with a documented public-code decision-line discrepancy: ``PFSkNN.m`` @ 9e04b32
  line 65 computes ``C(mode(NN(1:k)))``, the mode of the neighbours' row *indices* (all distinct,
  so the smallest index wins), whereas Algorithm 1, line 10 takes the most repetitive class
  *label* among the k nearest neighbours. This module follows the paper.

Intentional protocol correction for all three: the public code (and Definitions 33-34 of the
PFS-kNN paper) min-max normalise training and test data together. Here ranges are fitted on the
training data only and held-out values are clipped to [0, 1], so no test-fold information enters
model fitting. Constant training columns map to 1, as in MATLAB ``normalise``.

Tie rules (deterministic): equal distances are resolved in favour of the lower training-row index
(MATLAB ``sort``/``min``/``max`` are stable); equal vote counts are resolved in favour of the
smallest class label (MATLAB ``mode``). None of this is a bit-for-bit MATLAB equivalence claim.
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


def _absolute_pearson_or_nan(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation per column; NaN where it is undefined (constant column).

    Mirrors MATLAB ``abs(corr(x, y, 'Type', 'Pearson', 'Rows', 'complete'))``. Labels must be
    numeric, as in the MATLAB code; for two classes |r| does not depend on the label coding.
    """

    x_centered = x - x.mean(axis=0, keepdims=True)
    y_float = y.astype(float)
    y_centered = y_float - y_float.mean()
    numerator = x_centered.T @ y_centered
    denominator = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2))
    defined = (np.ptp(x, axis=0) != 0) & (denominator != 0)
    correlations = np.full(numerator.shape, np.nan)
    correlations[defined] = numerator[defined] / denominator[defined]
    return np.abs(correlations)


def _absolute_pearson_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """FPFS-kNN weights: |Pearson r| with undefined values set to 0 (``fw(isnan(fw))=0``)."""

    return np.nan_to_num(_absolute_pearson_or_nan(x, y), nan=0.0)


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
    """Minimal estimator interface shared by the three soft classifiers."""

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
    """Leakage-controlled adaptation of ``FPFSkNN.m`` (five metrics, per-class k-mean, vote).

    ``k`` is capped at the size of each training class, as the MATLAB code averages over all
    available rows when a class has fewer than ``k``.
    """

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
    """Leakage-controlled adaptation of ``IFPIFSHC.m`` (Hamming pseudo-similarity, 1-NN).

    The MATLAB similarity is ``1 - d / (2n)`` with ``d`` the L1 distance between the weighted
    (mu, nu, pi) embeddings; the nearest training row by ``d`` is therefore the most similar one.
    """

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
        correlation = _absolute_pearson_or_nan(x_raw, y_array)
        # As in IFPIFSHC.m: ifwP(isnan(ifwP))=0 sets both mu and nu weights of an undefined
        # correlation to 0, so its pi weight is 1.
        self.mu_weight_ = np.nan_to_num(1.0 - (1.0 - correlation) ** self.lambda1, nan=0.0)
        self.nu_weight_ = np.nan_to_num(
            (1.0 - correlation) ** (self.lambda1 * (self.lambda1 + 1.0)), nan=0.0
        )
        self.pi_weight_ = 1.0 - self.mu_weight_ - self.nu_weight_
        self.embedded_train_ = self._embed(self.x_train_scaled_)
        return self

    def training_distances(self, x: np.ndarray) -> np.ndarray:
        """L1 distances between weighted embeddings (test rows x training rows)."""
        return cdist(self._embed(self._transform_test(x)), self.embedded_train_, "cityblock")

    def nearest_training_index(self, x: np.ndarray) -> np.ndarray:
        """Index of the most similar training row; ties go to the lower index."""
        embedded = self._embed(self._transform_test(x))
        nearest: list[np.ndarray] = []
        for start in range(0, embedded.shape[0], self.chunk_size):
            distance = cdist(
                embedded[start : start + self.chunk_size],
                self.embedded_train_,
                metric="cityblock",
            )
            nearest.append(distance.argmin(axis=1))
        return np.concatenate(nearest)

    def predict(self, x: np.ndarray) -> np.ndarray:
        nearest = self.nearest_training_index(x)
        return self.y_train_[nearest]


class PFSKNNClassifier(_BaseSoftClassifier):
    """Paper-concordant PFS-kNN (Memiş, 2023, Algorithm 1) with majority voting over labels.

    Distances follow Proposition 7, ``d = ((1/3) * sum(|dmu|^p + |deta|^p + |dnu|^p +
    |dpi|^p))^(1/p)``, as in ``pfsMd`` of ``PFSkNN.m``. Neighbours are ranked by the unscaled
    Minkowski distance, which orders rows identically (monotone transform). The predicted class
    is the most repetitive class label among the k nearest neighbours (Algorithm 1, line 10),
    not the MATLAB line ``C(mode(NN(1:k)))``; see the module docstring.
    """

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

    def pfs_distances(self, x: np.ndarray) -> np.ndarray:
        """Proposition 7 distances (test rows x training rows), including the 1/3 scale."""
        embedded = self._embed(self._transform_test(x))
        unscaled = cdist(embedded, self.embedded_train_, metric="minkowski", p=self.p)
        return unscaled / 3.0 ** (1.0 / self.p)

    def kneighbors(self, x: np.ndarray) -> np.ndarray:
        """Indices of the k nearest training rows, nearest first.

        ``k`` is capped at the number of training rows. A stable sort resolves equal distances
        in favour of the lower training-row index, like MATLAB ``sort``.
        """
        embedded = self._embed(self._transform_test(x))
        effective_k = min(self.k, self.embedded_train_.shape[0])
        neighbours: list[np.ndarray] = []
        for start in range(0, embedded.shape[0], self.chunk_size):
            distance = cdist(
                embedded[start : start + self.chunk_size],
                self.embedded_train_,
                metric="minkowski",
                p=self.p,
            )
            neighbours.append(np.argsort(distance, axis=1, kind="stable")[:, :effective_k])
        return np.concatenate(neighbours)

    def predict(self, x: np.ndarray) -> np.ndarray:
        neighbours = self.kneighbors(x)
        encoded_train = np.searchsorted(self.classes_, self.y_train_)
        encoded_neighbours = encoded_train[neighbours]
        # Most repetitive label; equal counts go to the smallest label (MATLAB ``mode``).
        winner_indices = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=self.classes_.size).argmax(),
            1,
            encoded_neighbours,
        )
        return self.classes_[winner_indices]
