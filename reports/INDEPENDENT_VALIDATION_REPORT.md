# Independent Validation and Pilot Experiment Report

Date: 2026-09-28  
Dataset: UCI Taiwanese Bankruptcy Prediction  
Dataset DOI: <https://doi.org/10.24432/C5004D>

## 1. Status

The uploaded scaffold is **GO for data preparation** and **pilot-only for modelling**.
The original ZIP contained no implemented classifier. This independent copy adds classical
baselines and three source-derived soft classifiers, but the Python ports still require a
cross-language numerical fixture before they can be described as implementation-equivalent to
the MATLAB programs.

## 2. Reproducible data evidence

- Source ZIP SHA-256: `c346f5ad2618cb198e7ed8306cf2f31fe3bb2ec60acdbbe1736788d50f269aac`
- Raw shape: 6819 rows × 96 columns
- Clean shape: 6819 rows × 93 columns
- Class counts: 6599 non-bankrupt, 220 bankrupt
- Bankrupt prevalence: 3.2263%
- Missing cells: 0
- Exact duplicate rows: 0
- Constant explanatory columns: `Net Income Flag` only
- Exact duplicate explanatory pairs:
  - `Current Liabilities/Liability` = `Current Liability to Liability`
  - `Current Liabilities/Equity` = `Current Liability to Equity`
- `Liability-Assets Flag` is not constant: 6811 zeros and 8 ones; it is retained.

The five stored test folds each contain exactly 44 bankrupt observations. Four folds contain
1364 observations and the fifth contains 1363.

## 3. Quality gates

- Uploaded snapshot plus pilot additions: **50 passed**
- Remote foundation commit `cae36066fa9f13c517437cc35ec904605c4da9dd`: **52 passed**
- Merged remote foundation plus pilot additions: pending the included GitHub Actions workflow;
  no merged-pass claim is made before that run completes.
- Ruff on both independently checked trees: **all checks passed**
- Dependency check in the isolated validation environment: **no broken requirements**
- Python: 3.12.14
- pandas: 3.0.6
- numpy: 2.3.5
- scipy: 1.17.0
- scikit-learn: 1.8.0

## 4. Protocol

All models use the same stored five stratified folds. No hyperparameter search was conducted.
Scaling is fitted only on each training fold. Results are fold mean ± sample standard deviation.

The public MATLAB demos for FPFS-kNN, IFPIFS-HC and PFS-kNN concatenate training and test data
before min-max normalisation. The primary Python pilot deliberately does not reproduce that step:
minima and maxima are learned on the training fold and held-out values are clipped to [0,1]. This
is an intentional leakage-control adaptation and must be disclosed in the paper.

## 5. Classical baseline pilot

| Model | Balanced accuracy | Positive precision | Positive recall | Positive F1 | MCC | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy-prior | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.032 ± 0.000 |
| Logistic-balanced | 0.836 ± 0.029 | 0.174 ± 0.012 | 0.800 ± 0.069 | 0.285 ± 0.016 | 0.334 ± 0.022 | 0.893 ± 0.037 | 0.353 ± 0.075 |
| SVM-RBF-balanced | 0.787 ± 0.039 | 0.185 ± 0.006 | 0.673 ± 0.090 | 0.290 ± 0.012 | 0.315 ± 0.026 | 0.914 ± 0.016 | 0.327 ± 0.036 |
| RandomForest-balanced | 0.571 ± 0.030 | 0.652 ± 0.149 | 0.145 ± 0.061 | 0.229 ± 0.069 | 0.290 ± 0.051 | 0.940 ± 0.022 | 0.429 ± 0.046 |

Interpretation: threshold-free ranking and default-threshold classification answer different
questions. Random Forest ranks bankrupt firms well but predicts few positives at the default
threshold. Balanced logistic regression detects far more bankrupt firms but with low precision.

## 6. Soft-classifier pilot

| Model | Balanced accuracy | Positive precision | Positive recall | Positive F1 | MCC | Mean fold runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| FPFS-kNN (k=3, Pearson) | 0.605 ± 0.020 | 0.449 ± 0.098 | 0.218 ± 0.038 | 0.293 ± 0.055 | 0.297 ± 0.063 | 10.33 ± 0.23 |
| IFPIFS-HC (λ₁=5, λ₂=0.5) | 0.615 ± 0.030 | 0.346 ± 0.081 | 0.245 ± 0.059 | 0.286 ± 0.064 | 0.271 ± 0.067 | 1.62 ± 0.12 |
| PFS-kNN (k=3, λ=0.5, p=5) | 0.550 ± 0.010 | 0.424 ± 0.071 | 0.105 ± 0.020 | 0.167 ± 0.029 | 0.198 ± 0.032 | 45.05 ± 6.37 |

These are pilot results under fixed source-demo hyperparameters, not optimized estimates. A poor
minority recall is substantive evidence under the defined protocol, but it is not proof that an
algorithm family is generally inferior.

## 7. Primary source code provenance

- FPFS-kNN MATLAB repository: <https://github.com/sametmemis/FPFS-kNN>, inspected commit
  `776f69c66e252c3c6b23ff55e2708a8b9ad206f3`
- IFPIFS-HC MATLAB repository: <https://github.com/sametmemis/IFPIFS-HC>, inspected commit
  `a9d3093a0ee61a4e7f7cffde11929ec5f3922d21`
- PFS-kNN MATLAB repository: <https://github.com/sametmemis/PFS-kNN>, inspected commit
  `9e04b32fc48142123e0c9c9d1d5f55922b7b7cde`

No original MATLAB source file is redistributed. The Python code is a vectorised reimplementation
of the published transformations and decision rules.

## 8. What may be stated now

It is defensible to state that, in an initial five-fold leakage-controlled pilot on a severely
imbalanced bankruptcy dataset, the three fixed-parameter soft classifiers exhibited limited
bankrupt-class recall, while class-weighted conventional baselines improved sensitivity. It is
also defensible to state that performance rankings depend materially on the chosen metric.

## 9. What may not yet be stated

- The Python implementations are bit-for-bit identical to MATLAB.
- Any method is statistically superior on the basis of one dataset and five folds.
- The displayed results are optimized or final generalisation estimates.
- High accuracy alone demonstrates useful bankruptcy detection.
- Threshold-dependent values and ROC/PR ranking values are interchangeable.

## 10. Minimum remaining work before abstract submission

1. Review the Python equations line-by-line against the papers/public MATLAB functions.
2. Add one small frozen input/output fixture produced by MATLAB, if MATLAB access becomes available.
3. Decide whether the abstract is explicitly a fixed-parameter pilot or includes training-only
   parameter selection.
4. Freeze the result files and their hashes.
5. Draft the ASES abstract using only the verified claims above.
