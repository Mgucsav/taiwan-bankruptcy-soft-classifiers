# Independent Real-Data Check

Date: 2026-09-28. Script: `scripts/independent_real_data_check.py` (numpy, scipy, pandas only).

The check re-derives everything from the official UCI file without using the project code, then
compares with the canonical results in `reports/pilot_results/`. The numbers below come from a
local run on the real data (CPython 3.14.6, numpy 2.5.0, pandas 3.0.3, scipy 1.18.0) using the
same algorithm code as the committed script, with the fold assignments of CI run 36434283378.

## A. Data provenance

- Official ZIP re-downloaded from UCI: SHA-256
  `c346f5ad2618cb198e7ed8306cf2f31fe3bb2ec60acdbbe1736788d50f269aac`, identical to CI.
- Raw data 6819 × 96; classes 6599 / 220; 0 missing cells; 0 duplicate rows.
- Constant column: `Net Income Flag`. Identical pairs: `Current Liabilities/Liability` =
  `Current Liability to Liability`, `Current Liabilities/Equity` = `Current Liability to Equity`.
- Clean data 6819 × 93, equal to the output of the project's cleaning code. `Liability-Assets Flag`
  is retained and the target is not among the features.

## B. Folds

The stored fold file is in row order, its targets equal the clean target, and every fold holds
44 bankrupt firms (1364 / 1364 / 1364 / 1364 / 1363 firms). The fold file of the first CI run
(36422198527) and of run 36434283378 are identical.

## C. Independent re-computation of the soft classifiers

FPFS-kNN, IFPIFS-HC and PFS-kNN were re-implemented line by line from the public MATLAB functions
(and, for PFS-kNN, Algorithm 1 of the paper), with training-fold-only min-max scaling. Metrics were
computed from explicit formulas.

| Model | Predictions differing from production code | Count mismatches vs canonical | Max. metric difference vs canonical |
|---|---:|---:|---:|
| FPFS-kNN (k=3, Pearson) | 0 / 6819 | 0 | 0.0 |
| IFPIFS-HC (λ₁=5, λ₂=0.5) | 0 / 6819 | 0 | 0.0 |
| PFS-kNN (k=3, λ=0.5, p=5) | 0 / 6819 | 0 | 0.0 |

Model labels and parameters match the author demo calls in the MATLAB files:
`FPFSkNN(…,3,'Pearson')`, `IFPIFSHC(…,5,0.5)` (λ₁ → feature weights, λ₂ → data) and
`PFSkNN(…,3,0.5,5)`.

## D. PFS-kNN: paper rule versus the public MATLAB decision line

`PFSkNN.m` line 65, `C(mode(NN(1:k)))`, takes the mode of the neighbours' row indices. These are
distinct, so it returns the label of the neighbour that appears earliest in the training file.
Algorithm 1 takes the most repetitive class label.

| Rule | Predicted bankrupt (all folds) | Balanced acc. | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Algorithm 1, label majority (used) | 55 | 0.550 | 0.424 | 0.105 | 0.167 | 0.198 |
| MATLAB line 65 | 261 | 0.588 | 0.177 | 0.209 | 0.191 | 0.163 |

The rules disagree for 230 of 6819 firms (3.4 %). The UCI file is not randomly ordered with respect
to the target: its first six rows are bankrupt firms, 77.3 % of all bankrupt firms lie in the first
half of the file, and the mean row index is 2378 for bankrupt versus 3443 for non-bankrupt firms.
The MATLAB line therefore favours bankrupt neighbours, and its result depends on the order of the
file rather than on the method. A direct test (shuffling the rows and re-running the MATLAB line) has
not been performed.

## Limits and open points

- The classical baselines were not re-run locally (scikit-learn is blocked by Windows App Control on
  the development machine). They are verified by CI and by count/metric consistency checks only.
- The independent code was written by the same author as the adaptation, from the same reading of
  the MATLAB code and the PFS-kNN paper. No MATLAB run was performed. The FPFS-kNN and IFPIFS-HC
  papers were not compared, only their MATLAB code.
- `reports/pilot_results/baseline_protocol.json` lists `pr_auc, mcc, balanced_accuracy,
  recall_positive` as primary metrics, whereas the README and the validation report name balanced
  accuracy, precision, recall, F1 and MCC. This inconsistency is not yet resolved.
- FPFS-AC and IFPIFSC are not implemented yet.
