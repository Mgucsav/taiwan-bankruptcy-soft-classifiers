"""Independent formula fixtures for FPFS-kNN, IFPIFS-HC and PFS-kNN.

The expected values below were produced by the pure-Python reference functions in this file
(loops and ``math`` only; no numpy/scipy, no production code). They transcribe the public MATLAB
functions (FPFSkNN.m @ 776f69c, IFPIFSHC.m @ a9d3093) and, for PFS-kNN, Definitions 35-36,
Proposition 7 and Algorithm 1 of Memiş (2023), Electronics 12(19), 4129.

These are *independent formula fixtures*, not MATLAB outputs: no MATLAB run was performed, so no
bit-for-bit MATLAB equivalence is claimed. The fixture is already min-max normalised (every
training column spans [0, 1]), so training-fold scaling is the identity here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from taiwan_soft_classifiers.soft_classifiers import (
    FPFSKNNClassifier,
    IFPIFSHCClassifier,
    PFSKNNClassifier,
)

RTOL = 1e-10
ATOL = 1e-12

X_TRAIN = [
    [0.0, 0.2, 1.0],
    [0.3, 0.0, 0.6],
    [0.1, 0.5, 0.8],
    [0.4, 0.3, 0.9],
    [0.8, 1.0, 0.0],
    [1.0, 0.7, 0.3],
    [0.6, 0.9, 0.2],
]
Y_TRAIN = [0, 0, 0, 0, 1, 1, 1]
X_TEST = [
    [0.2, 0.3, 0.7],
    [0.7, 0.6, 0.4],
    [0.45, 0.5, 0.5],
]
# Source parameters used in the experiments.
K = 3
LAMBDA1, LAMBDA2 = 5.0, 0.5
PFS_LAMBDA, PFS_P = 0.5, 5.0


# ---------------------------------------------------------------------------
# Pure-Python reference formulas (independent of the production code)
# ---------------------------------------------------------------------------
def ref_abs_pearson(column: list[float], labels: list[int]) -> float:
    """abs(corr(x, y, 'Type', 'Pearson')); NaN for a constant column."""
    n = len(column)
    mx = sum(column) / n
    my = sum(labels) / n
    num = sum((a - mx) * (b - my) for a, b in zip(column, labels, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in column) * sum((b - my) ** 2 for b in labels))
    if max(column) == min(column) or den == 0:
        return math.nan
    return abs(num / den)


def ref_columns(rows: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*rows, strict=True)]


def ref_mode(values: list[int]) -> int:
    """MATLAB ``mode``: most frequent value, smallest value on ties."""
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    return min(v for v, c in counts.items() if c == best)


def ref_fpfs(x_train, y_train, x_test, k):
    w = [ref_abs_pearson(col, y_train) for col in ref_columns(x_train)]
    w = [0.0 if math.isnan(v) else v for v in w]  # fw(isnan(fw))=0
    classes = sorted(set(y_train))
    g = []  # g[test][class][metric], mean of the k smallest per metric (G in FPFSkNN.m)
    for t in x_test:
        per_class = []
        for r in classes:
            rows = [row for row, lab in zip(x_train, y_train, strict=True) if lab == r]
            values: list[list[float]] = [[], [], [], [], []]
            for b in rows:
                diffs = [abs(w[j] * t[j] - w[j] * b[j]) for j in range(len(t))]
                values[0].append(sum(diffs))  # fpfsd1 Hamming
                values[1].append(max(diffs))  # fpfsd2 Chebyshev
                values[2].append(math.sqrt(sum(d**2 for d in diffs)))  # fpfsd3 Euclidean
                values[3].append(max(diffs))  # fpfsd5 Hamming-Hausdorff (single row)
                values[4].append(sum(d**3 for d in diffs) ** (1 / 3))  # fpfsd6 Minkowski p=3
            kk = min(k, len(rows))
            per_class.append([sum(sorted(v)[:kk]) / kk for v in values])
        g.append(per_class)
    winners = []
    for per_class in g:
        row = []
        for m in range(5):
            column = [per_class[r][m] for r in range(len(classes))]
            row.append(column.index(min(column)))  # MATLAB min: first index
        winners.append(row)
    predictions = [classes[ref_mode(row)] for row in winners]
    return w, g, winners, predictions


def ref_ifpifs(x_train, y_train, x_test, lambda1, lambda2):
    r = [ref_abs_pearson(col, y_train) for col in ref_columns(x_train)]
    mu_w = [0.0 if math.isnan(v) else 1 - (1 - v) ** lambda1 for v in r]
    nu_w = [0.0 if math.isnan(v) else (1 - v) ** (lambda1 * (lambda1 + 1)) for v in r]
    pi_w = [1 - a - b for a, b in zip(mu_w, nu_w, strict=True)]

    def components(row):
        mu = [1 - (1 - x) ** lambda2 for x in row]
        nu = [(1 - x) ** (lambda2 * (lambda2 + 1)) for x in row]
        pi = [1 - a - b for a, b in zip(mu, nu, strict=True)]
        return mu, nu, pi

    test_components = [components(t) for t in x_test]
    train_components = [components(b) for b in x_train]
    n = len(x_test[0])
    m = 2  # two-row ifpifs-matrices: weight row + sample row
    similarity = []
    for mu_a, nu_a, pi_a in test_components:
        row = []
        for mu_b, nu_b, pi_b in train_components:
            d = 0.0
            for j in range(n):
                d += abs(mu_w[j] * mu_a[j] - mu_w[j] * mu_b[j])
                d += abs(nu_w[j] * nu_a[j] - nu_w[j] * nu_b[j])
                d += abs(pi_w[j] * pi_a[j] - pi_w[j] * pi_b[j])
            row.append(1 - d / (2 * (m - 1) * n))  # ifpifsHs
        similarity.append(row)
    selected = [row.index(max(row)) for row in similarity]  # MATLAB max: first index
    predictions = [y_train[i] for i in selected]
    return (mu_w, nu_w, pi_w), test_components, similarity, selected, predictions


def ref_pfs(x_train, y_train, x_test, k, lam, p):
    def components(row):
        mu = [1 - (1 - x) ** lam for x in row]  # Definition 35/36
        eta = [x / lam for x in row]
        nu = [(1 - x) ** (lam * (lam + 1)) for x in row]
        pi = [1 - a - b for a, b in zip(mu, nu, strict=True)]  # pi = 1 - (mu + nu)
        return mu, eta, nu, pi

    test_components = [components(t) for t in x_test]
    train_components = [components(b) for b in x_train]
    distances = []
    for a in test_components:
        row = []
        for b in train_components:
            total = 0.0
            for comp_a, comp_b in zip(a, b, strict=True):
                total += sum(abs(u - v) ** p for u, v in zip(comp_a, comp_b, strict=True))
            row.append((total / 3) ** (1 / p))  # Proposition 7
        distances.append(row)
    neighbours = [
        sorted(range(len(row)), key=lambda j, row=row: (row[j], j))[:k] for row in distances
    ]
    labels = [[y_train[j] for j in nn] for nn in neighbours]
    majority = [ref_mode(lab) for lab in labels]  # Algorithm 1, line 10
    return test_components, distances, neighbours, labels, majority


# ---------------------------------------------------------------------------
# Frozen values (output of the reference functions above)
# ---------------------------------------------------------------------------
FPFS_WEIGHTS = [0.8798826901281197, 0.8870071077479607, 0.9203484446124429]
# [test][class][Hamming, Chebyshev, Euclidean, Hamming-Hausdorff, Minkowski-3]
FPFS_CLASS_DISTANCES = [
    [
        [0.38786533592340183, 0.2091910809321563, 0.25600576826429466,
         0.2091910809321563, 0.2304763578063662],
        [1.5214194789273525, 0.6267847759933275, 0.8985224788400599,
         0.6267847759933275, 0.7645807370492235],
    ],
    [
        [1.014412631329401, 0.5067693670106231, 0.6357580287381794,
         0.5067693670106231, 0.5601785865324841],
        [0.5979303141637129, 0.2994021057359338, 0.3824653617254276,
         0.2994021057359338, 0.3409534836274179],
    ],
    [
        [0.6137064035613308, 0.37320062442126645, 0.4321377911355852,
         0.37320062442126645, 0.3997574415166181],
        [0.9399776959232419, 0.43297084832529054, 0.575096602008748,
         0.43297084832529054, 0.5021443434479762],
    ],
]  # fmt: skip
FPFS_METRIC_WINNERS = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
FPFS_PREDICTIONS = [0, 1, 0]

IFPIFS_MU_WEIGHT = [0.9999749949350917, 0.9999815814419758, 0.9999967939425204]
IFPIFS_NU_WEIGHT = [2.444375473236607e-28, 3.90421530412004e-29, 1.0859950002241907e-33]
IFPIFS_PI_WEIGHT = [2.5005064908323327e-05, 1.8418558024202447e-05, 3.2060574796100028e-06]
# [test][mu, nu, pi][feature]
IFPIFS_TEST_COMPONENTS = [
    [
        [0.10557280900008414, 0.16333997346592444, 0.4522774424948338],
        [0.8458970107524514, 0.7652855797503654, 0.4053600464421104],
        [0.04853018024746447, 0.07137444678371019, 0.1423625110630558],
    ],
    [
        [0.4522774424948338, 0.3675444679663241, 0.2254033307585166],
        [0.4053600464421104, 0.5029733718731741, 0.6817316198804996],
        [0.1423625110630558, 0.12948216016050174, 0.09286504936098383],
    ],
    [
        [0.2583801512904337, 0.2928932188134524, 0.2928932188134524],
        [0.6386633830041155, 0.5946035575013605, 0.5946035575013605],
        [0.1029564657054508, 0.11250322368518706, 0.11250322368518706],
    ],
]
IFPIFS_SIMILARITY = [
    [0.881490139703429, 0.9490271141708476, 0.9526140629679134, 0.941446217334657,
     0.710645452715341, 0.754620622620667, 0.8118175225424508],
    [0.7518612497554401, 0.8668978497860826, 0.8661689923547969, 0.8517601567192112,
     0.8402740388071357, 0.8842487833713844, 0.9132027394380114],
    [0.8078667870504674, 0.9229034080681484, 0.9221745506368628, 0.9077657150012771,
     0.7842685847675787, 0.828243553539559, 0.8854409071056096],
]  # fmt: skip
IFPIFS_SELECTED = [2, 6, 1]
IFPIFS_PREDICTIONS = [0, 1, 0]

# [test][mu, eta, nu, pi][feature]
PFS_TEST_COMPONENTS = [
    [
        [0.10557280900008414, 0.16333997346592444, 0.4522774424948338],
        [0.4, 0.6, 1.4],
        [0.8458970107524514, 0.7652855797503654, 0.4053600464421104],
        [0.04853018024746447, 0.07137444678371019, 0.1423625110630558],
    ],
    [
        [0.4522774424948338, 0.3675444679663241, 0.2254033307585166],
        [1.4, 1.2, 0.8],
        [0.4053600464421104, 0.5029733718731741, 0.6817316198804996],
        [0.1423625110630558, 0.12948216016050174, 0.09286504936098383],
    ],
    [
        [0.2583801512904337, 0.2928932188134524, 0.2928932188134524],
        [0.9, 1.0, 1.0],
        [0.6386633830041155, 0.5946035575013605, 0.5946035575013605],
        [0.1029564657054508, 0.11250322368518706, 0.11250322368518706],
    ],
]
PFS_DISTANCES = [
    [0.5483415136804262, 0.48347403161046926, 0.3262509116740689, 0.3738175731584267,
     1.362323108147922, 1.3230681398451025, 1.0566552045112434],
    [1.2378719571940784, 0.9912904103311991, 0.9916079664576029, 0.8342191839227092,
     0.7663513952789414, 0.5407559985768318, 0.5016147648322632],
    [0.9153640623466135, 0.8052817277324255, 0.6091148559983243, 0.6543207838286574,
     0.9594439181755215, 0.918588785696266, 0.6783346286490541],
]  # fmt: skip
PFS_NEIGHBOURS = [[2, 3, 1], [6, 5, 4], [2, 3, 6]]
PFS_NEIGHBOUR_LABELS = [[0, 0, 0], [1, 1, 1], [0, 0, 1]]
PFS_MAJORITY = [0, 1, 0]
PFS_PREDICTIONS = [0, 1, 0]


def assert_close(actual, expected) -> None:
    np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=RTOL, atol=ATOL)


def as_arrays():
    return np.array(X_TRAIN), np.array(Y_TRAIN), np.array(X_TEST)


# ---------------------------------------------------------------------------
# The reference formulas reproduce the frozen values
# ---------------------------------------------------------------------------
def test_reference_formulas_reproduce_frozen_values() -> None:
    w, g, winners, predictions = ref_fpfs(X_TRAIN, Y_TRAIN, X_TEST, K)
    assert_close(w, FPFS_WEIGHTS)
    assert_close(g, FPFS_CLASS_DISTANCES)
    assert winners == FPFS_METRIC_WINNERS
    assert predictions == FPFS_PREDICTIONS

    weights, comps, sim, selected, preds = ref_ifpifs(X_TRAIN, Y_TRAIN, X_TEST, LAMBDA1, LAMBDA2)
    assert_close(weights, [IFPIFS_MU_WEIGHT, IFPIFS_NU_WEIGHT, IFPIFS_PI_WEIGHT])
    assert_close(comps, IFPIFS_TEST_COMPONENTS)
    assert_close(sim, IFPIFS_SIMILARITY)
    assert selected == IFPIFS_SELECTED
    assert preds == IFPIFS_PREDICTIONS

    comps, dist, nn, labels, majority = ref_pfs(X_TRAIN, Y_TRAIN, X_TEST, K, PFS_LAMBDA, PFS_P)
    assert_close(comps, PFS_TEST_COMPONENTS)
    assert_close(dist, PFS_DISTANCES)
    assert nn == PFS_NEIGHBOURS
    assert labels == PFS_NEIGHBOUR_LABELS
    assert majority == PFS_MAJORITY


# ---------------------------------------------------------------------------
# The production classes match the frozen values
# ---------------------------------------------------------------------------
def test_fpfs_knn_independent_formula_fixture() -> None:
    x_train, y_train, x_test = as_arrays()
    model = FPFSKNNClassifier(k=K).fit(x_train, y_train)
    assert_close(model.feature_weights_, FPFS_WEIGHTS)
    weighted = model._transform_test(x_test) * model.feature_weights_
    distances = model._class_metric_distances(weighted)  # (test, class, metric)
    assert_close(distances, FPFS_CLASS_DISTANCES)
    np.testing.assert_array_equal(np.argmin(distances, axis=1), FPFS_METRIC_WINNERS)
    np.testing.assert_array_equal(model.predict(x_test), FPFS_PREDICTIONS)


def test_ifpifs_hc_independent_formula_fixture() -> None:
    x_train, y_train, x_test = as_arrays()
    model = IFPIFSHCClassifier(lambda1=LAMBDA1, lambda2=LAMBDA2).fit(x_train, y_train)
    assert_close(model.mu_weight_, IFPIFS_MU_WEIGHT)
    assert_close(model.nu_weight_, IFPIFS_NU_WEIGHT)
    assert_close(model.pi_weight_, IFPIFS_PI_WEIGHT)

    n = x_test.shape[1]
    embedded = model._embed(model._transform_test(x_test))  # [mu*w_mu, nu*w_nu, pi*w_pi]
    comps = np.array(IFPIFS_TEST_COMPONENTS)
    weights = [IFPIFS_MU_WEIGHT, IFPIFS_NU_WEIGHT, IFPIFS_PI_WEIGHT]
    for c in range(3):
        assert_close(embedded[:, c * n : (c + 1) * n], comps[:, c, :] * np.array(weights[c]))

    similarity = 1.0 - model.training_distances(x_test) / (2.0 * n)
    assert_close(similarity, IFPIFS_SIMILARITY)
    np.testing.assert_array_equal(model.nearest_training_index(x_test), IFPIFS_SELECTED)
    np.testing.assert_array_equal(model.predict(x_test), IFPIFS_PREDICTIONS)


def test_pfs_knn_independent_formula_fixture() -> None:
    x_train, y_train, x_test = as_arrays()
    model = PFSKNNClassifier(k=K, lambda_value=PFS_LAMBDA, p=PFS_P).fit(x_train, y_train)
    n = x_test.shape[1]
    embedded = model._embed(model._transform_test(x_test))  # [mu, eta, nu, pi]
    comps = np.array(PFS_TEST_COMPONENTS)
    for c in range(4):
        assert_close(embedded[:, c * n : (c + 1) * n], comps[:, c, :])
    assert_close(model.pfs_distances(x_test), PFS_DISTANCES)
    neighbours = model.kneighbors(x_test)
    np.testing.assert_array_equal(neighbours, PFS_NEIGHBOURS)
    np.testing.assert_array_equal(y_train[neighbours], PFS_NEIGHBOUR_LABELS)
    np.testing.assert_array_equal(model.predict(x_test), PFS_MAJORITY)
    np.testing.assert_array_equal(model.predict(x_test), PFS_PREDICTIONS)


# ---------------------------------------------------------------------------
# PFS-kNN: paper Algorithm 1 versus the public MATLAB decision line
# ---------------------------------------------------------------------------
def matlab_decision_line(neighbour_indices: list[int], labels: np.ndarray) -> int:
    """``PredictedClass = C(mode(NN(1:k)))`` from PFSkNN.m line 65 (0-based here)."""
    return int(labels[ref_mode(list(neighbour_indices))])


def test_pfs_knn_uses_label_majority_not_mode_of_row_indices() -> None:
    # One feature spanning [0, 1]. For x = 0.51 the three nearest rows are 4, 5 and 0 with
    # labels 1, 1 and 0. The mode of the (distinct) row indices is the smallest index, 0,
    # whose label is 0; the most repetitive label (Algorithm 1, line 10) is 1.
    x_train = np.array([[0.45], [0.0], [0.1], [1.0], [0.5], [0.56]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x_test = np.array([[0.51]])
    model = PFSKNNClassifier(k=3, lambda_value=0.5, p=5).fit(x_train, y_train)

    neighbours = model.kneighbors(x_test)[0]
    assert set(neighbours.tolist()) == {0, 4, 5}
    majority = ref_mode(y_train[neighbours].tolist())
    matlab_line = matlab_decision_line(neighbours.tolist(), y_train)
    assert majority == 1
    assert matlab_line == 0
    assert majority != matlab_line
    assert model.predict(x_test).tolist() == [majority]


@pytest.mark.parametrize("row", range(len(X_TEST)))
def test_fixture_rows_predicted_alone_match_batch(row: int) -> None:
    x_train, y_train, x_test = as_arrays()
    for model in (
        FPFSKNNClassifier(k=K),
        IFPIFSHCClassifier(lambda1=LAMBDA1, lambda2=LAMBDA2),
        PFSKNNClassifier(k=K, lambda_value=PFS_LAMBDA, p=PFS_P),
    ):
        model.fit(x_train, y_train)
        batch = model.predict(x_test)
        alone = model.predict(x_test[row : row + 1])
        assert alone[0] == batch[row]
