# Third-Party Notices

This repository's own Python code is released under the MIT License (see `LICENSE`).
The works below are **not** covered by that license. No third-party source code
(including the MATLAB implementations listed below) is copied into this repository;
the classifiers will be re-implemented independently in Python from the published papers.

## Dataset

**Taiwanese Bankruptcy Prediction** — UCI Machine Learning Repository

- Page: <https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction>
- DOI: <https://doi.org/10.24432/C5004D>
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Associated article: D. Liang, C.-C. Lu, C.-F. Tsai, G.-A. Shih (2016). Financial ratios and
  corporate governance indicators in bankruptcy prediction: A comprehensive study.
  *European Journal of Operational Research*, 252(2), 561–572.
  <https://doi.org/10.1016/j.ejor.2016.01.012>

The data are downloaded at run time by `scripts/download_data.py` and are not redistributed
in this repository. Derived files (`data/processed/…`, `artifacts/…`) are adaptations of the
dataset: the constant column `Net Income Flag` and two exact duplicate columns are removed,
and column-name whitespace is stripped.

## Reference MATLAB implementations (not included)

The repositories below did not state a license at the time of writing; they are cited as
scholarly references only.

| Method | Repository | Academic work |
|---|---|---|
| FPFS-kNN | <https://github.com/sametmemis/FPFS-kNN> | S. Memiş, S. Enginoğlu, U. Erkan (2022). Fuzzy parameterized fuzzy soft *k*-nearest neighbor classifier. *Neurocomputing*, 500, 351–378. <https://doi.org/10.1016/j.neucom.2022.05.041> |
| FPFS-AC | <https://github.com/sametmemis/FPFS-AC> | S. Memiş, S. Enginoğlu, U. Erkan (2022). A new classification method using soft decision-making based on an aggregation operator of fuzzy parameterized fuzzy soft matrices. *Turkish Journal of Electrical Engineering and Computer Sciences*, 30(3), 1165–1180. <https://doi.org/10.55730/1300-0632.3816> |
| IFPIFS-HC | <https://github.com/sametmemis/IFPIFS-HC> | S. Memiş, B. Arslan, T. Aydın, S. Enginoğlu, Ç. Camcı (2021). A classification method based on Hamming pseudo-similarity of intuitionistic fuzzy parameterized intuitionistic fuzzy soft matrices. *Journal of New Results in Science*, 10(2), 59–76. <https://dergipark.org.tr/en/pub/jnrs/issue/64701/981326> |
| IFPIFSC | <https://github.com/sametmemis/IFPIFSC> | S. Memiş, B. Arslan, T. Aydın, S. Enginoğlu, Ç. Camcı (2023). Distance and similarity measures of intuitionistic fuzzy parameterized intuitionistic fuzzy soft matrices and their applications to data classification in supervised learning. *Axioms*, 12(5), 463. <https://doi.org/10.3390/axioms12050463> |
| PFS-kNN | <https://github.com/sametmemis/PFS-kNN> | S. Memiş (2023). Picture fuzzy soft matrices and application of their distance measures to supervised learning: Picture fuzzy soft *k*-nearest neighbor (PFS-kNN). *Electronics*, 12(19), 4129. <https://doi.org/10.3390/electronics12194129> |

## Python dependencies

Installed from PyPI under their own licenses (not vendored): NumPy, SciPy, scikit-learn,
pandas, Matplotlib, seaborn, joblib, and for development pytest, pytest-cov and Ruff.
