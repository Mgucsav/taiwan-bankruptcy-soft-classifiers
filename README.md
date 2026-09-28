# Corporate Bankruptcy Prediction via Generalized Soft Matrix-Based Classification Algorithms on Taiwan Stock Exchange Data

## 1. Exact English project title

**Corporate Bankruptcy Prediction via Generalized Soft Matrix-Based Classification Algorithms on Taiwan Stock Exchange Data**

## 2. Research objective

To evaluate soft-matrix-based classifiers — fuzzy parameterized fuzzy soft, intuitionistic
fuzzy parameterized intuitionistic fuzzy soft and picture fuzzy soft matrices — for corporate
bankruptcy prediction on Taiwan Stock Exchange data, under a single, leakage-free and fully
reproducible cross-validation protocol.

## 3. Dataset

UCI Machine Learning Repository, *Taiwanese Bankruptcy Prediction*.

- Page: <https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction>
- DOI: <https://doi.org/10.24432/C5004D>
- License: CC BY 4.0
- Firms listed on the Taiwan Stock Exchange, 1999–2009, bankruptcy defined by TSE business
  regulations (Liang et al., 2016).

Raw data contract (checked by `scripts/prepare_data.py`; any violation raises an error):

| Property | Expected |
|---|---|
| Rows | 6819 |
| Columns | 96 (95 explanatory + target `Bankrupt?`) |
| Target classes | {0, 1}; 6599 non-bankrupt, 220 bankrupt |
| Missing cells | 0 |
| Exact duplicate rows | 0 |

Cleaning decisions are based on numerical checks of the data, never on column names:

1. Header names with leading/trailing whitespace are reported (`data_quality.json`) and
   stripped; names must remain unique afterwards.
2. Constant features (a single unique value) are detected; the target is excluded.
3. Exactly identical feature pairs are detected by value; in each group the first column in
   file order is kept and later copies are dropped. The target is never compared.
4. The detected structure must match the expected one — exactly one constant column
   (`Net Income Flag`) and two duplicate features — and `Liability-Assets Flag` must not be
   among the drops. Otherwise cleaning stops, drops nothing and prints every constant column,
   every identical pair, the proposed drops and the shapes before/after.

Expected result: 6819 rows × 93 columns (92 features + target), class counts unchanged, no
missing values. The expected duplicates are `Current Liability to Liability`
(= `Current Liabilities/Liability`) and `Current Liability to Equity`
(= `Current Liabilities/Equity`).

## 4. Planned classifiers

Not implemented yet; each will be implemented in Python in a separate task.

| Classifier | Reference |
|---|---|
| FPFS-kNN | Memiş, Enginoğlu & Erkan (2022), *Neurocomputing* |
| FPFS-AC | Memiş, Enginoğlu & Erkan (2022), *Turk. J. Elec. Eng. & Comp. Sci.* |
| IFPIFS-HC | Memiş, Arslan, Aydın, Enginoğlu & Camcı (2021), *J. New Results Sci.* |
| IFPIFSC | Memiş, Arslan, Aydın, Enginoğlu & Camcı (2023), *Axioms* |
| PFS-kNN | Memiş (2023), *Electronics* |

## 5. Class imbalance warning

Only 220 of 6819 firms (≈3.23 %) are bankrupt — about 30 non-bankrupt firms per bankrupt firm.
A classifier that always predicts "non-bankrupt" reaches ≈96.8 % accuracy, so accuracy alone
is not informative. Results must be reported with imbalance-aware metrics (e.g. recall and
precision of the bankrupt class, F1, balanced accuracy, MCC, ROC-AUC/PR-AUC), and any class
balancing must follow the leakage rules below.

## 6. Leakage prevention protocol

**Scaling, feature selection and class balancing are fitted exclusively on each training fold.**

- The cleaning step only removes columns that are constant or exact duplicates over the whole
  file — a property of the schema, not a statistic learned from data values.
- No min-max normalisation, feature selection or under/over-sampling is applied during data
  preparation.
- Folds are fixed once (`StratifiedKFold`, `n_splits=5`, `shuffle=True`, `random_state=42`)
  and saved in `artifacts/fold_assignments.csv`, so every classifier sees identical splits.
- For each fold, normalisation ranges, selected features and resampling are derived from the
  four training folds only, then applied unchanged to the held-out test fold.
- `artifacts/data_profile.csv` is descriptive only and must never be used to fit a model.

## 7. Repository structure

```
taiwan-bankruptcy-soft-classifiers/
├── README.md
├── LICENSE                     # MIT (our code only)
├── CITATION.cff
├── THIRD_PARTY_NOTICES.md      # dataset (CC BY 4.0) and reference works
├── pyproject.toml
├── .devcontainer/              # GitHub Codespaces (Python 3.12)
├── data/
│   ├── README.md
│   ├── raw/                    # UCI ZIP, data.csv, download metadata (git-ignored)
│   └── processed/              # taiwan_bankruptcy_clean.csv (git-ignored)
├── artifacts/                  # small reproducible outputs (tracked)
│   ├── data_quality.json
│   ├── data_profile.csv
│   └── fold_assignments.csv
├── scripts/
│   ├── download_data.py
│   └── prepare_data.py
├── src/taiwan_soft_classifiers/
│   ├── config.py               # paths, RANDOM_STATE = 42, data contracts
│   ├── data.py                 # download, safe extraction, loading, cleaning, profiling
│   ├── validation.py           # contract checks (raise DataValidationError)
│   └── splitting.py            # fixed stratified folds
└── tests/
```

## 8. Installation

Requires Python ≥ 3.12, < 3.15. The reference environment is the GitHub Codespaces
dev container in `.devcontainer/` (Python 3.12, Debian Bookworm), which installs the project
automatically.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Core dependencies: numpy ≥ 2.3.4, scipy ≥ 1.16.1, scikit-learn ≥ 1.8, pandas, matplotlib,
seaborn, joblib. Development: pytest, pytest-cov, ruff.

> **Windows note:** if Smart App Control / App Control for Business is enabled, it may block
> the unsigned compiled extensions of pandas or scikit-learn
> (`ImportError: DLL load failed ... blocked by App Control policy`). Run the project in an
> environment where these wheels are allowed (e.g. WSL, a Linux/macOS machine or CI).

## 9. Data download

```bash
python scripts/download_data.py            # reuses an existing valid ZIP
python scripts/download_data.py --force    # download again
python scripts/download_data.py --expected-sha256 <hex>   # optionally pin the digest
```

The ZIP is streamed to a temporary file and moved into place atomically only after a complete
download, so an interrupted download never leaves a corrupt `data/raw/…zip`. Only `data.csv`
is extracted (no `extractall`; path-traversal and size checks are applied). The SHA-256 of the
ZIP and the CSV and the UTC download time are stored in `data/raw/download_metadata.json`.

## 10. Data preparation

```bash
python scripts/prepare_data.py
```

Validates the raw contract, cleans the data, validates the clean contract, creates and
validates the folds, and writes:

- `data/processed/taiwan_bankruptcy_clean.csv`
- `artifacts/data_quality.json` — source, DOI, license, download time, ZIP SHA-256, shapes,
  class counts/ratios, missing and duplicate counts, removed columns, raw numeric min/max
- `artifacts/data_profile.csv` — per feature: dtype, unique count, min, 1 %, median, mean,
  std, 99 %, max, zero ratio, missing ratio
- `artifacts/fold_assignments.csv` — `row_id` (0-based row position), `target`, `fold` (0–4)

## 11. Tests

```bash
pytest -q
ruff check .
```

Tests run on the locally downloaded real data and never access the network; data-dependent
tests are skipped with an explanatory message if the data have not been downloaded. Failure
cases are produced by perturbing copies of the real data.

## 12. Current project status

- [x] Project infrastructure, data download, validation, cleaning, fixed folds, tests
- [ ] FPFS-kNN
- [ ] FPFS-AC
- [ ] IFPIFS-HC
- [ ] IFPIFSC
- [ ] PFS-kNN
- [ ] Experiments and reporting

No classifier has been implemented and no model results exist yet.

## 13. Reproducibility

- `RANDOM_STATE = 42` for every random operation.
- Folds are stored in `artifacts/fold_assignments.csv` and re-checked by the tests.
- The source ZIP's SHA-256 is recorded in `artifacts/data_quality.json`; rerun with
  `--expected-sha256` to require the same file.
- All paths are relative to the repository (`pathlib.Path`); no credentials are used.
- Note: `data_quality.json` contains the download timestamp, so it changes if the data are
  downloaded again even when the data are identical.

## 14. Academic references

- Liang, D., Lu, C.-C., Tsai, C.-F., & Shih, G.-A. (2016). Financial ratios and corporate
  governance indicators in bankruptcy prediction: A comprehensive study. *European Journal of
  Operational Research*, 252(2), 561–572. <https://doi.org/10.1016/j.ejor.2016.01.012>
- Taiwanese Bankruptcy Prediction [Dataset]. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C5004D>
- Memiş, S., Enginoğlu, S., & Erkan, U. (2022). Fuzzy parameterized fuzzy soft *k*-nearest
  neighbor classifier. *Neurocomputing*, 500, 351–378.
  <https://doi.org/10.1016/j.neucom.2022.05.041>
- Memiş, S., Enginoğlu, S., & Erkan, U. (2022). A new classification method using soft
  decision-making based on an aggregation operator of fuzzy parameterized fuzzy soft matrices.
  *Turkish Journal of Electrical Engineering and Computer Sciences*, 30(3), 1165–1180.
  <https://doi.org/10.55730/1300-0632.3816>
- Memiş, S., Arslan, B., Aydın, T., Enginoğlu, S., & Camcı, Ç. (2021). A classification method
  based on Hamming pseudo-similarity of intuitionistic fuzzy parameterized intuitionistic fuzzy
  soft matrices. *Journal of New Results in Science*, 10(2), 59–76.
- Memiş, S., Arslan, B., Aydın, T., Enginoğlu, S., & Camcı, Ç. (2023). Distance and similarity
  measures of intuitionistic fuzzy parameterized intuitionistic fuzzy soft matrices and their
  applications to data classification in supervised learning. *Axioms*, 12(5), 463.
  <https://doi.org/10.3390/axioms12050463>
- Memiş, S. (2023). Picture fuzzy soft matrices and application of their distance measures to
  supervised learning: Picture fuzzy soft *k*-nearest neighbor (PFS-kNN). *Electronics*,
  12(19), 4129. <https://doi.org/10.3390/electronics12194129>

## 15. Code and data licenses

- **Code:** the Python code in this repository is released under the MIT License (`LICENSE`).
- **Data:** the UCI Taiwanese Bankruptcy Prediction dataset is licensed separately under
  CC BY 4.0 and is not redistributed here. See `THIRD_PARTY_NOTICES.md`.
- The reference MATLAB implementations are cited only; none of their code is included.
