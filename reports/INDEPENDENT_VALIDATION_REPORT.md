# Independent Validation and Pilot Experiment Report

Date: 2026-09-28<br>
Dataset: UCI Taiwanese Bankruptcy Prediction<br>
Dataset DOI: <https://doi.org/10.24432/C5004D>

## 1. Status

The data pipeline is validated on the real UCI data in GitHub Actions (Linux, Python 3.12).
Modelling is at **fixed-parameter pilot** stage: a preliminary comparison with fixed,
source-based parameters, reported as descriptive results from one dataset and five folds.

- FPFS-kNN and IFPIFS-HC are leakage-controlled Python adaptations of the transformations and
  decision rules published in the authors' MATLAB code.
- PFS-kNN is a paper-concordant Python implementation with majority voting over the class labels
  of the k nearest neighbours (Memiş, 2023, Algorithm 1), with a documented public-code
  decision-line discrepancy (Section 7).
- The implementations are checked against **independent formula fixtures**. No MATLAB run was
  performed, so no bit-for-bit MATLAB equivalence is claimed.

## 2. Reproducible data evidence

- Source ZIP SHA-256: `c346f5ad2618cb198e7ed8306cf2f31fe3bb2ec60acdbbe1736788d50f269aac`
- Raw shape: 6819 rows × 96 columns
- Clean shape: 6819 rows × 93 columns
- Class counts: 6599 non-bankrupt, 220 bankrupt (prevalence 3.2263%)
- Missing cells: 0; exact duplicate rows: 0
- Constant explanatory columns: `Net Income Flag` only (dropped)
- Exactly identical explanatory pairs (later copy dropped):
  - `Current Liabilities/Liability` = `Current Liability to Liability`
  - `Current Liabilities/Equity` = `Current Liability to Equity`
- `Liability-Assets Flag` is neither constant nor a duplicate (6811 zeros, 8 ones) and is
  retained.

The five stored test folds each contain exactly 44 bankrupt observations; four folds contain
1364 observations and the fifth 1363.

## 3. Quality gates and environment

Canonical run: GitHub Actions run `36422198527` (commit `b7f7a96`), all steps passed.

| Package | Version (install log of run 36422198527) |
|---|---|
| Python | 3.12.14 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |

Run `36425263045` confirmed the same versions in `artifacts/environment.json`, which
`scripts/write_environment.py` fills from the running environment.

**Pinned environment.** CI runs on `ubuntu-24.04` with CPython `3.12.14`. It installs numpy,
pandas, scipy, scikit-learn, joblib and threadpoolctl at fixed versions from
`constraints-ci.txt` (`pip install -e ".[dev]" -c constraints-ci.txt`). `pyproject.toml` keeps
lower bounds only; pytest and Ruff are not pinned because they do not produce numerical
results.

The workflow also sets `PYTHONHASHSEED=42` and restricts OpenBLAS, OpenMP, MKL and numexpr to
one thread. `OPENBLAS_CORETYPE` is not forced. Instead, `artifacts/environment.json` records:

- the CPU (`lscpu` summary, `platform.machine()`);
- `numpy.show_config()`;
- `threadpoolctl.threadpool_info()`, including the OpenBLAS core type in use;
- the thread environment variables.

**Cross-runner observation.** Runs `36425263045` (Azure westus3) and `36431187367` (Azure
westcentralus) used identical package versions and runner images. Their predicted labels,
confusion counts and threshold-based metrics were identical for every model. The soft-classifier
results and the SVM, random forest and dummy results were identical in every column. Only the
Logistic-balanced ROC-AUC and PR-AUC differed, by at most 7.5e-4 per fold and 2.6e-4 in the
means. Logistic regression probability-score rankings showed cross-runner numerical variation
below 1e-3 while predicted labels, confusion counts and threshold-based metrics remained
identical.

**Canonical-results gate (reporting-level reproducibility).** After the experiments,
`scripts/verify_reproducibility.py` compares every file in `reports/pilot_results/` with the
newly produced file in `artifacts/`. The results are claimed to be numerically stable within the
declared tolerance, not bit-for-bit identical on every CPU.

| Fields | Rule |
|---|---|
| Model name, fold, sample and class counts, predicted positives, TP/TN/FP/FN, other text or integer fields | exact equality |
| Balanced accuracy, precision, recall, F1, MCC; every soft-classifier metric; ROC-AUC and PR-AUC of all models except Logistic-balanced | rtol = atol = 1e-12 |
| Logistic-balanced ROC-AUC and PR-AUC, and their fold mean/std | rtol = 0, atol = 1e-3 |
| Runtime (`elapsed_seconds*`) | ignored |
| JSON protocol files | equal as parsed objects |

Any violation fails the workflow. Because of the 1e-3 tolerance, the Logistic-balanced ROC-AUC
and PR-AUC are reported to two decimals. The workflow additionally runs the baseline experiment
three times on the same runner (`scripts/diagnose_baseline_determinism.py`) and records in
`artifacts/baseline_determinism.json` whether the Logistic AUCs are bit-for-bit identical
within one runner.

The canonical files are unchanged. Their SHA-256 digests are frozen in
`reports/pilot_results/SHA256SUMS.txt` and checked by the test suite.

Random forest threshold-dependent results differed between scikit-learn 1.8.0 (the environment
of the first independent pilot) and 1.9.1 by a small margin that is nevertheless larger than
rounding error. The final pilot therefore uses the results of the CI environment of run
36422198527, whose package versions are now pinned (see above). The baseline tables below and
`reports/pilot_results/baseline_*.csv` are taken from that run. The soft-classifier results of
that run were identical to the first pilot and have stayed identical in every later CI run.

## 4. Protocol

All models use the same stored five stratified folds. No hyperparameter search was conducted
and no parameter was chosen on test folds. Scaling is fitted only on each training fold.
Results are fold mean ± sample standard deviation.

The public MATLAB demos (and Definitions 33-34 of the PFS-kNN paper) min-max normalise training
and test data together. The Python implementations deliberately do not reproduce that step:
minima and maxima are learned on the training fold, held-out values are clipped to [0, 1], and a
constant training column maps to 1 (as in MATLAB `normalise`). This is an intentional
leakage-control adaptation and must be disclosed.

Fixed source parameters: FPFS-kNN k = 3 with Pearson weights; IFPIFS-HC λ₁ = 5, λ₂ = 0.5;
PFS-kNN k = 3, λ = 0.5, p = 5.

Each fold result includes `true_positive`, `true_negative`, `false_positive` and
`false_negative`. In the canonical files they were derived exactly from `n_test`,
`positive_test`, `predicted_positive` and recall, reproduce every stored precision, balanced
accuracy and MCC to 1e-12, and are re-checked by the test suite. The soft classifiers produce
labels, not continuous scores, so no ROC-AUC or PR-AUC is reported for them.

## 5. Classical baseline pilot (canonical CI results)

| Model | Balanced accuracy | Positive precision | Positive recall | Positive F1 | MCC | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy-prior | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.032 ± 0.000 |
| Logistic-balanced | 0.836 ± 0.029 | 0.174 ± 0.012 | 0.800 ± 0.069 | 0.285 ± 0.016 | 0.334 ± 0.022 | 0.89 ± 0.04ᵃ | 0.35 ± 0.07ᵃ |
| SVM-RBF-balanced | 0.787 ± 0.039 | 0.185 ± 0.006 | 0.673 ± 0.090 | 0.290 ± 0.012 | 0.315 ± 0.026 | 0.914 ± 0.016 | 0.327 ± 0.036 |
| RandomForest-balanced | 0.571 ± 0.030 | 0.611 ± 0.110 | 0.145 ± 0.061 | 0.227 ± 0.070 | 0.281 ± 0.052 | 0.942 ± 0.020 | 0.430 ± 0.046 |

ᵃ Two decimals: these values are reproducible within an absolute tolerance of 1e-3 across CI
runners (Section 3). All other entries are reproduced to 1e-12.

Threshold-free ranking (ROC-AUC, PR-AUC) and default-threshold classification answer different
questions. In this pilot the random forest ranks bankrupt firms well but predicts few positives
at the default threshold; balanced logistic regression flags many more bankrupt firms at low
precision.

## 6. Soft-classifier pilot

| Model | Balanced accuracy | Positive precision | Positive recall | Positive F1 | MCC | Mean fold runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| FPFS-kNN (k=3, Pearson) | 0.605 ± 0.020 | 0.449 ± 0.098 | 0.218 ± 0.038 | 0.293 ± 0.055 | 0.297 ± 0.063 | 10.33 ± 0.23 |
| IFPIFS-HC (λ₁=5, λ₂=0.5) | 0.615 ± 0.030 | 0.346 ± 0.081 | 0.245 ± 0.059 | 0.286 ± 0.064 | 0.271 ± 0.067 | 1.62 ± 0.12 |
| PFS-kNN (k=3, λ=0.5, p=5) | 0.550 ± 0.010 | 0.424 ± 0.071 | 0.105 ± 0.020 | 0.167 ± 0.029 | 0.198 ± 0.032 | 45.05 ± 6.37 |

Runtimes are from the first independent pilot and depend on the machine; they are not part of
any equality check. These are descriptive pilot results under fixed source parameters, not
optimised or final estimates, and they do not show that any method is statistically superior.

## 7. Source provenance and implementation verification

| Method | Public code (inspected commit) | Paper |
|---|---|---|
| FPFS-kNN | <https://github.com/sametmemis/FPFS-kNN> `776f69c66e252c3c6b23ff55e2708a8b9ad206f3` | Neurocomputing 500 (2022) 351-378 |
| IFPIFS-HC | <https://github.com/sametmemis/IFPIFS-HC> `a9d3093a0ee61a4e7f7cffde11929ec5f3922d21` | J. New Results Sci. 10(2) (2021) 59-76 |
| PFS-kNN | <https://github.com/sametmemis/PFS-kNN> `9e04b32fc48142123e0c9c9d1d5f55922b7b7cde` | Electronics 12(19) (2023) 4129 |

No original MATLAB source file is redistributed.

**PFS-kNN decision line.** `PFSkNN.m` line 65 reads `PredictedClass(i,1)=C(mode(NN(1:k)));`,
where `NN` holds the training-row indices sorted by distance. Because these indices are
distinct, `mode` returns the smallest index among the k nearest neighbours, and the prediction
is the label of that single row. Algorithm 1 of the paper (line 10) instead takes "the most
repetitive class label in the considered k-nearest neighbor". The Python implementation follows
the paper. `tests/test_soft_verification.py` contains a regression case in which the two rules
disagree (mode of indices → class 0, label majority → class 1) and checks that Python returns the
label majority. The component definitions (Definitions 35-36), the refusal degree
π = 1 − (μ + ν) and the Minkowski distance with the 1/3 factor (Proposition 7) agree between the
paper, the MATLAB code and the Python implementation. On the real data the two rules disagree for
about 3.4 % of the firms and change the PFS-kNN results materially, because bankrupt firms are
concentrated near the top of the UCI file (see `reports/INDEPENDENT_REAL_DATA_CHECK.md`).

**IFPIFS-HC undefined correlations.** As in `IFPIFSHC.m` (`ifwP(isnan(ifwP))=0`), a feature
whose Pearson correlation is undefined (constant training column) receives zero μ and ν weights,
hence π weight 1. With training-fold scaling such a column is constant for all rows, so this
rule does not change predictions.

**Tie rules.** Equal distances resolve to the lower training-row index (MATLAB `sort`/`min`/
`max` are stable); equal vote counts resolve to the smallest class label (MATLAB `mode`).

**Independent formula fixtures.** A 7 × 3 training matrix and three test rows (already min-max
normalised) are evaluated by pure-Python transcriptions of the formulas (no production code),
frozen, and compared with the implementations at `rtol = 1e-10`, `atol = 1e-12`:

- FPFS-kNN: absolute Pearson weights, per-class k-mean of the five metrics, per-metric winners,
  prediction.
- IFPIFS-HC: μ, ν, π components, weight components, Hamming pseudo-similarity, selected training
  row, prediction.
- PFS-kNN: membership, neutrality, non-membership and refusal components, Proposition 7
  distance, k neighbour indices and labels, majority class, prediction.

Separate tests cover training-only min-max fitting, clipping, constant columns, independence
from other test rows, non-finite inputs, unfitted models, parameter bounds, k larger than the
class or training size, non-0/1 labels and the tie rules.

## 8. What may be stated now

In an initial five-fold, leakage-controlled pilot on a severely imbalanced bankruptcy dataset,
the three fixed-parameter soft classifiers showed limited bankrupt-class recall, while the
class-weighted conventional baselines reached higher sensitivity. Performance rankings depend
materially on the chosen metric.

## 9. What may not be stated

- The Python implementations are bit-for-bit identical to the MATLAB programs.
- PFS-kNN reproduces the published MATLAB decision line.
- Any method is statistically superior on the basis of one dataset and five folds.
- The displayed results come from optimised models or are final generalisation estimates.
- High accuracy alone demonstrates useful bankruptcy detection.
- Threshold-dependent values and ROC/PR ranking values are interchangeable.

## 10. Remaining work

1. If MATLAB access becomes available, run the three public functions on the frozen fixture
   (with the joint normalisation switched to training-only) and add the outputs as a separate
   MATLAB fixture.
2. Decide whether the study stays a fixed-parameter pilot or adds training-fold-only parameter
   selection (nested within the training folds).

Completed: the CI environment is pinned (`constraints-ci.txt`, Ubuntu 24.04, CPython 3.12.14),
and the canonical result files are frozen by SHA-256 and protected by the CI comparison gate.
