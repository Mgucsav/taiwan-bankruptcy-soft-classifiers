"""Training-fold normalisation, input validation, parameter bounds, labels and tie rules."""

from __future__ import annotations

import numpy as np
import pytest

from taiwan_soft_classifiers.soft_classifiers import (
    FPFSKNNClassifier,
    IFPIFSHCClassifier,
    PFSKNNClassifier,
    _TrainingMinMax,
)

X_TRAIN = np.array(
    [
        [2.0, 10.0, 5.0],
        [4.0, 30.0, 5.0],
        [3.0, 20.0, 5.0],
        [8.0, 60.0, 5.0],
        [9.0, 50.0, 5.0],
        [7.0, 70.0, 5.0],
    ]
)
Y_TRAIN = np.array([0, 0, 0, 1, 1, 1])
X_TEST = np.array([[3.5, 25.0, 5.0], [8.5, 55.0, 5.0], [5.5, 40.0, 6.0]])


def all_models():
    return [
        FPFSKNNClassifier(k=3),
        IFPIFSHCClassifier(lambda1=5, lambda2=0.5),
        PFSKNNClassifier(k=3, lambda_value=0.5, p=5),
    ]


MODEL_IDS = ["FPFS-kNN", "IFPIFS-HC", "PFS-kNN"]


# ---------------------------------------------------------------------------
# _TrainingMinMax
# ---------------------------------------------------------------------------
def test_minmax_learns_bounds_from_training_data_only() -> None:
    scaler = _TrainingMinMax.fit(X_TRAIN)
    np.testing.assert_array_equal(scaler.minimum, X_TRAIN.min(axis=0))
    np.testing.assert_array_equal(scaler.maximum, X_TRAIN.max(axis=0))
    scaler.transform(np.array([[-1e9, 1e9, -1e9]]))
    np.testing.assert_array_equal(scaler.minimum, X_TRAIN.min(axis=0))
    np.testing.assert_array_equal(scaler.maximum, X_TRAIN.max(axis=0))


def test_minmax_clips_values_outside_training_range() -> None:
    scaler = _TrainingMinMax.fit(X_TRAIN)
    below = scaler.transform(np.array([[1.0, 0.0, 5.0]]))
    above = scaler.transform(np.array([[100.0, 700.0, 5.0]]))
    np.testing.assert_array_equal(below[0, :2], [0.0, 0.0])
    np.testing.assert_array_equal(above[0, :2], [1.0, 1.0])
    inside = scaler.transform(np.array([[5.5, 40.0, 5.0]]))
    np.testing.assert_allclose(inside[0, :2], [0.5, 0.5])


def test_minmax_constant_training_column_maps_to_one() -> None:
    # MATLAB ``normalise``: max == min -> ones. Test values differing from the constant
    # are also mapped to 1 because the training span is zero.
    scaler = _TrainingMinMax.fit(X_TRAIN)
    transformed = scaler.transform(np.array([[3.0, 20.0, 5.0], [3.0, 20.0, -40.0]]))
    np.testing.assert_array_equal(transformed[:, 2], [1.0, 1.0])
    np.testing.assert_array_equal(scaler.transform(X_TRAIN)[:, 2], np.ones(len(X_TRAIN)))


# ---------------------------------------------------------------------------
# Classifiers: leakage control and batch independence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", all_models(), ids=MODEL_IDS)
def test_extreme_test_values_do_not_change_training_state(model) -> None:
    model.fit(X_TRAIN, Y_TRAIN)
    minimum, maximum = model.scaler_.minimum.copy(), model.scaler_.maximum.copy()
    train_scaled = model.x_train_scaled_.copy()
    model.predict(np.array([[1e12, -1e12, 1e12], [-1e12, 1e12, -1e12]]))
    np.testing.assert_array_equal(model.scaler_.minimum, minimum)
    np.testing.assert_array_equal(model.scaler_.maximum, maximum)
    np.testing.assert_array_equal(model.x_train_scaled_, train_scaled)
    np.testing.assert_array_equal(model.scaler_.minimum, X_TRAIN.min(axis=0))
    np.testing.assert_array_equal(model.scaler_.maximum, X_TRAIN.max(axis=0))


@pytest.mark.parametrize("model", all_models(), ids=MODEL_IDS)
def test_out_of_range_test_values_are_clipped(model) -> None:
    model.fit(X_TRAIN, Y_TRAIN)
    scaled = model._transform_test(np.array([[-50.0, 1000.0, 5.0]]))
    np.testing.assert_array_equal(scaled, [[0.0, 1.0, 1.0]])
    # Values beyond the training range predict exactly like the clipped boundary values.
    np.testing.assert_array_equal(
        model.predict(np.array([[-50.0, 1000.0, 5.0]])),
        model.predict(np.array([[2.0, 70.0, 5.0]])),
    )


@pytest.mark.parametrize("model", all_models(), ids=MODEL_IDS)
def test_prediction_does_not_depend_on_other_test_rows(model) -> None:
    model.fit(X_TRAIN, Y_TRAIN)
    extreme = np.array([[1e9, -1e9, 1e9]])
    batch = np.vstack([X_TEST, extreme])
    together = model.predict(batch)
    alone = np.concatenate([model.predict(row.reshape(1, -1)) for row in batch])
    np.testing.assert_array_equal(together, alone)
    np.testing.assert_array_equal(model.predict(X_TEST), together[: len(X_TEST)])


@pytest.mark.parametrize("model", all_models(), ids=MODEL_IDS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_raise(model, bad: float) -> None:
    broken = X_TRAIN.copy()
    broken[0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        model.fit(broken, Y_TRAIN)
    model.fit(X_TRAIN, Y_TRAIN)
    broken_test = X_TEST.copy()
    broken_test[1, 1] = bad
    with pytest.raises(ValueError, match="non-finite"):
        model.predict(broken_test)


@pytest.mark.parametrize("model", all_models(), ids=MODEL_IDS)
def test_predict_before_fit_raises(model) -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(X_TEST)


@pytest.mark.parametrize("model", all_models(), ids=MODEL_IDS)
def test_feature_count_mismatch_and_single_class_raise(model) -> None:
    with pytest.raises(ValueError, match="two classes"):
        model.fit(X_TRAIN, np.zeros(len(X_TRAIN), dtype=int))
    model.fit(X_TRAIN, Y_TRAIN)
    with pytest.raises(ValueError, match="features"):
        model.predict(X_TEST[:, :2])


# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, -1])
def test_fpfs_knn_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        FPFSKNNClassifier(k=k)


@pytest.mark.parametrize(("lambda1", "lambda2"), [(0, 0.5), (-1, 0.5), (5, 0), (5, -0.5)])
def test_ifpifs_hc_rejects_non_positive_lambdas(lambda1: float, lambda2: float) -> None:
    with pytest.raises(ValueError, match="lambda1 and lambda2 must be positive"):
        IFPIFSHCClassifier(lambda1=lambda1, lambda2=lambda2)


@pytest.mark.parametrize(
    ("k", "lam", "p"),
    [(0, 0.5, 5), (-2, 0.5, 5), (3, 0, 5), (3, -0.5, 5), (3, 0.5, 0), (3, 0.5, -1)],
)
def test_pfs_knn_rejects_non_positive_parameters(k: int, lam: float, p: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        PFSKNNClassifier(k=k, lambda_value=lam, p=p)


def test_fpfs_knn_k_larger_than_class_uses_whole_class() -> None:
    # FPFSkNN.m averages over all rows of a class that has fewer than k rows.
    capped = FPFSKNNClassifier(k=100).fit(X_TRAIN, Y_TRAIN)
    exact = FPFSKNNClassifier(k=3).fit(X_TRAIN, Y_TRAIN)
    weighted = capped._transform_test(X_TEST) * capped.feature_weights_
    np.testing.assert_allclose(
        capped._class_metric_distances(weighted), exact._class_metric_distances(weighted)
    )
    np.testing.assert_array_equal(capped.predict(X_TEST), exact.predict(X_TEST))


def test_pfs_knn_k_larger_than_training_set_is_capped() -> None:
    x_train = X_TRAIN[:5]
    y_train = Y_TRAIN[:5]  # three 0s, two 1s
    model = PFSKNNClassifier(k=100).fit(x_train, y_train)
    neighbours = model.kneighbors(X_TEST)
    assert neighbours.shape == (len(X_TEST), len(x_train))
    for row in neighbours:
        assert sorted(row.tolist()) == list(range(len(x_train)))
    # All training rows vote, so the majority class of the training set is predicted.
    np.testing.assert_array_equal(model.predict(X_TEST), [0, 0, 0])


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("index", range(3), ids=MODEL_IDS)
@pytest.mark.parametrize(("negative", "positive"), [(1, 2), (-1, 5), (7, 3)])
def test_labels_other_than_zero_one(index: int, negative: int, positive: int) -> None:
    # FPFS-kNN and IFPIFS-HC need numeric labels (Pearson weights, as in MATLAB); for two
    # classes |r| is invariant to the label coding, so predictions only get relabelled.
    reference = all_models()[index].fit(X_TRAIN, Y_TRAIN).predict(X_TEST)
    recoded = np.where(Y_TRAIN == 1, positive, negative)
    model = all_models()[index].fit(X_TRAIN, recoded)
    expected = np.where(reference == 1, positive, negative)
    np.testing.assert_array_equal(model.predict(X_TEST), expected)


def test_pfs_knn_accepts_non_numeric_labels() -> None:
    labels = np.where(Y_TRAIN == 1, "bankrupt", "healthy")
    reference = PFSKNNClassifier(k=3).fit(X_TRAIN, Y_TRAIN).predict(X_TEST)
    predicted = PFSKNNClassifier(k=3).fit(X_TRAIN, labels).predict(X_TEST)
    np.testing.assert_array_equal(predicted, np.where(reference == 1, "bankrupt", "healthy"))


def test_three_classes_are_supported() -> None:
    x_train = np.array([[0.0], [0.1], [0.5], [0.55], [0.9], [1.0]])
    y_train = np.array([1, 1, 2, 2, 3, 3])
    x_test = np.array([[0.05], [0.52], [0.95]])
    for model in all_models():
        model.fit(x_train, y_train)
        np.testing.assert_array_equal(model.predict(x_test), [1, 2, 3])


# ---------------------------------------------------------------------------
# IFPIFS-HC undefined correlation (IFPIFSHC.m: ifwP(isnan(ifwP))=0)
# ---------------------------------------------------------------------------
def test_ifpifs_hc_constant_training_column_weights() -> None:
    model = IFPIFSHCClassifier(lambda1=5, lambda2=0.5).fit(X_TRAIN, Y_TRAIN)
    assert model.mu_weight_[2] == 0.0
    assert model.nu_weight_[2] == 0.0
    assert model.pi_weight_[2] == 1.0


# ---------------------------------------------------------------------------
# Tie rules (documented and deterministic)
# ---------------------------------------------------------------------------
def test_ifpifs_hc_equal_distance_picks_lower_training_index() -> None:
    # Rows 0 and 1 are identical but labelled differently; MATLAB max() returns the first.
    x_train = np.array([[0.2, 0.2], [0.2, 0.2], [1.0, 0.0], [0.0, 1.0]])
    y_train = np.array([1, 0, 0, 1])
    model = IFPIFSHCClassifier().fit(x_train, y_train)
    test = np.array([[0.2, 0.2]])
    assert model.nearest_training_index(test).tolist() == [0]
    assert model.predict(test).tolist() == [1]
    swapped = IFPIFSHCClassifier().fit(x_train, np.array([0, 1, 0, 1]))
    assert swapped.predict(test).tolist() == [0]


def test_pfs_knn_equal_distance_picks_lower_training_index() -> None:
    # Rows 1 and 2 are identical, hence exactly equidistant from the query; with k = 2 the
    # stable ranking keeps the lower index (row 1), like MATLAB ``sort``.
    x_train = np.array([[0.5], [0.4], [0.4], [0.0], [1.0]])
    query = np.array([[0.5]])
    for labels in ([0, 1, 0, 0, 1], [0, 0, 1, 0, 1]):
        model = PFSKNNClassifier(k=2, lambda_value=0.5, p=5).fit(x_train, np.array(labels))
        distances = model.pfs_distances(query)[0]
        assert distances[1] == distances[2]
        assert model.kneighbors(query).tolist() == [[0, 1]]


def test_pfs_knn_equal_votes_pick_smallest_label() -> None:
    # k = 2 with one neighbour of each class: MATLAB ``mode`` returns the smaller label.
    x_train = np.array([[0.4], [0.7], [0.0], [1.0]])
    for labels, expected in (([5, 2, 5, 2], 2), ([2, 5, 2, 5], 2)):
        model = PFSKNNClassifier(k=2, lambda_value=1.0, p=1).fit(x_train, np.array(labels))
        assert model.predict(np.array([[0.55]])).tolist() == [expected]


def test_fpfs_knn_equal_class_distances_pick_smallest_label() -> None:
    # Two classes with identical training rows: every metric ties, argmin -> first class.
    x_train = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    y_train = np.array([4, 4, 9, 9])
    model = FPFSKNNClassifier(k=1).fit(x_train, np.array([4, 9, 4, 9]))
    assert model.predict(np.array([[0.3, 0.7]])).tolist() == [4]
    tied = FPFSKNNClassifier(k=2).fit(x_train, y_train)
    assert tied.predict(np.array([[0.3, 0.7]])).tolist() == [4]


@pytest.mark.parametrize("model", all_models(), ids=MODEL_IDS)
def test_repeated_predictions_are_identical(model) -> None:
    model.fit(X_TRAIN, Y_TRAIN)
    first = model.predict(X_TEST)
    for _ in range(3):
        np.testing.assert_array_equal(model.predict(X_TEST), first)
