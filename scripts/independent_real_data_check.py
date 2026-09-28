"""Independent end-to-end check of the three soft classifiers on the real UCI data.

Usage:
    python scripts/independent_real_data_check.py [--data-csv data/raw/data.csv]
        [--folds artifacts/fold_assignments.csv] [--output artifacts/independent_check.json]

Parts A-C use no project code:
  A. read the raw UCI CSV and re-derive shapes, class counts, constant and identical columns;
  B. check that the stored folds are aligned with the cleaned data;
  C. recompute FPFS-kNN, IFPIFS-HC and PFS-kNN line by line from the public MATLAB functions /
     the PFS-kNN paper (with training-fold-only min-max scaling), compute the metrics from
     explicit formulas and compare them with reports/pilot_results/; for PFS-kNN also evaluate
     the public MATLAB decision line C(mode(NN(1:k))) and relate it to the file row order.
Part D imports the production classes only to compare their predictions row by row.

Runtime: about 20-25 minutes on a laptop (the loops are deliberately literal, not vectorised).
Needs numpy, scipy and pandas only (no scikit-learn).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = "Bankrupt?"
NAMES = {
    "fpfs": "FPFS-kNN(k=3,Pearson)",
    "ifpifs": "IFPIFS-HC(lambda1=5,lambda2=0.5)",
    "pfs": "PFS-kNN(k=3,lambda=0.5,p=5)",
}


# --------------------------------------------------------------------------- helpers
def train_minmax(xtr: np.ndarray, x: np.ndarray) -> np.ndarray:
    mn, mx = xtr.min(0), xtr.max(0)
    span = mx - mn
    out = np.ones_like(x, dtype=float)
    nz = span != 0
    out[:, nz] = (x[:, nz] - mn[nz]) / span[nz]
    return np.clip(out, 0, 1)


def abs_pearson(xtr: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    r = np.empty(xtr.shape[1])
    for j in range(xtr.shape[1]):
        col = xtr[:, j]
        r[j] = np.nan if col.max() == col.min() else abs(np.corrcoef(col, ytr)[0, 1])
    return r


def matlab_mode(values: np.ndarray):
    """MATLAB ``mode``: most frequent value, smallest value on ties."""
    vals, counts = np.unique(values, return_counts=True)
    return vals[np.argmax(counts)]


# --------------------------------------------------------------------------- algorithms
def fpfs_knn(xtr, ytr, xte, k=3):
    fw = np.nan_to_num(abs_pearson(xtr, ytr), nan=0.0)  # fw(isnan(fw))=0
    tr, te = train_minmax(xtr, xtr), train_minmax(xtr, xte)
    classes = np.unique(ytr)
    pred = np.empty(len(te), dtype=ytr.dtype)
    for i, t in enumerate(te):
        g = np.empty((len(classes), 5))
        for r, c in enumerate(classes):
            d = np.abs(fw * t - fw * tr[ytr == c])  # |a(1,j)a(2,j) - b(1,j)b(2,j)|
            metrics = [
                d.sum(1),  # fpfsd1 Hamming
                d.max(1),  # fpfsd2 Chebyshev
                np.sqrt((d**2).sum(1)),  # fpfsd3 Euclidean
                d.max(1),  # fpfsd5 Hamming-Hausdorff (one sample row)
                ((d**3).sum(1)) ** (1 / 3),  # fpfsd6 Minkowski p=3
            ]
            kk = min(k, d.shape[0])
            g[r] = [np.sort(m)[:kk].mean() for m in metrics]
        pred[i] = classes[matlab_mode(g.argmin(0))]  # [~,h]=min(G); UClass(mode(h))
    return pred


def ifpifs_hc(xtr, ytr, xte, l1=5.0, l2=0.5):
    r = abs_pearson(xtr, ytr)
    mu_w = np.nan_to_num(1 - (1 - r) ** l1, nan=0.0)  # ifwP(isnan(ifwP))=0
    nu_w = np.nan_to_num((1 - r) ** (l1 * (l1 + 1)), nan=0.0)
    pi_w = 1 - mu_w - nu_w
    tr, te = train_minmax(xtr, xtr), train_minmax(xtr, xte)

    def comp(x):
        mu = 1 - (1 - x) ** l2
        nu = (1 - x) ** (l2 * (l2 + 1))
        return mu, nu, 1 - mu - nu

    tmu, tnu, tpi = comp(tr)
    n = xtr.shape[1]
    pred = np.empty(len(te), dtype=ytr.dtype)
    for i, t in enumerate(te):
        amu, anu, api = comp(t)
        d = (
            np.abs(mu_w * amu - mu_w * tmu).sum(1)
            + np.abs(nu_w * anu - nu_w * tnu).sum(1)
            + np.abs(pi_w * api - pi_w * tpi).sum(1)
        )
        s = 1 - d / (2 * 1 * n)  # ifpifsHs with m = 2
        pred[i] = ytr[np.argmax(s)]  # [~,w]=max(Sr): first maximum
    return pred


def pfs_knn(xtr, ytr, xte, k=3, lam=0.5, p=5.0):
    """Returns (Algorithm 1 majority vote, MATLAB line C(mode(NN(1:k)))) predictions."""
    tr, te = train_minmax(xtr, xtr), train_minmax(xtr, xte)

    def comp(x):
        mu = 1 - (1 - x) ** lam
        eta = x / lam
        nu = (1 - x) ** (lam * (lam + 1))
        return mu, eta, nu, 1 - mu - nu

    train_comp = comp(tr)
    paper = np.empty(len(te), dtype=ytr.dtype)
    matlab = np.empty(len(te), dtype=ytr.dtype)
    for i, t in enumerate(te):
        test_comp = comp(t)
        s = sum((np.abs(a - b) ** p).sum(1) for a, b in zip(test_comp, train_comp, strict=True))
        dm = (s / 3) ** (1 / p)  # pfsMd (Proposition 7)
        nn = np.argsort(dm, kind="stable")[:k]  # [~,NN]=sort(Dm)
        paper[i] = matlab_mode(ytr[nn])  # Algorithm 1, line 10
        matlab[i] = ytr[matlab_mode(nn)]  # PFSkNN.m line 65
    return paper, matlab


def metrics(yt: np.ndarray, yp: np.ndarray) -> dict[str, float | int]:
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_positive": tp + fp,
        "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
        "precision_positive": tp / (tp + fp) if tp + fp else 0.0,
        "recall_positive": tp / (tp + fn),
        "f1_positive": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "mcc": (tp * tn - fp * fn) / den if den else 0.0,
    }


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent real-data check.")
    parser.add_argument("--data-csv", type=Path, default=ROOT / "data/raw/data.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "artifacts/fold_assignments.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/independent_check.json")
    args = parser.parse_args(argv)
    out: dict[str, object] = {}

    # A. data
    raw = pd.read_csv(args.data_csv, float_precision="round_trip")
    raw.columns = [c.strip() for c in raw.columns]
    features = [c for c in raw.columns if c != TARGET]
    constant = [c for c in features if raw[c].nunique() == 1]
    identical: list[tuple[str, str]] = []
    for j, later in enumerate(features):
        for earlier in features[:j]:
            dropped = [d for _, d in identical]
            if earlier not in dropped and np.array_equal(raw[earlier], raw[later]):
                identical.append((earlier, later))
                break
    clean = raw.drop(columns=constant + [d for _, d in identical])
    labels_in_file_order = raw[TARGET].to_numpy()
    half = len(labels_in_file_order) // 2
    out["A_data"] = {
        "raw_shape": list(raw.shape),
        "clean_shape": list(clean.shape),
        "class_counts": {
            int(k): int(v) for k, v in raw[TARGET].value_counts().sort_index().items()
        },
        "missing_cells": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "constant_columns": constant,
        "identical_pairs": identical,
        "liability_assets_flag_kept": "Liability-Assets Flag" in clean.columns,
        "bankrupt_share_in_first_half_of_file": float(
            labels_in_file_order[:half].sum() / labels_in_file_order.sum()
        ),
        "first_10_labels_in_file_order": labels_in_file_order[:10].tolist(),
    }

    # B. folds
    folds = pd.read_csv(args.folds)
    x = clean.drop(columns=[TARGET]).to_numpy(dtype=float)
    y = clean[TARGET].to_numpy().astype(int)
    fold = folds["fold"].to_numpy()
    out["B_folds"] = {
        "row_id_in_order": bool((folds["row_id"].to_numpy() == np.arange(len(y))).all()),
        "targets_match": bool((folds["target"].to_numpy() == y).all()),
        "per_fold": {
            int(f): {"n": int((fold == f).sum()), "bankrupt": int(y[fold == f].sum())}
            for f in np.unique(fold)
        },
    }

    # C. independent algorithms
    pred = {k: np.empty(len(y), dtype=int) for k in (*NAMES, "pfs_matlab_line")}
    start = time.time()
    for f in np.unique(fold):
        test = fold == f
        xtr, ytr, xte = x[~test], y[~test], x[test]
        pred["fpfs"][test] = fpfs_knn(xtr, ytr, xte)
        pred["ifpifs"][test] = ifpifs_hc(xtr, ytr, xte)
        pred["pfs"][test], pred["pfs_matlab_line"][test] = pfs_knn(xtr, ytr, xte)
        print(f"independent fold {f} done ({time.time() - start:.0f}s)", flush=True)

    canonical = pd.read_csv(
        ROOT / "reports/pilot_results/soft_classifier_fold_results.csv",
        float_precision="round_trip",
    )
    comparison = {}
    for key, name in NAMES.items():
        mismatches, worst = [], 0.0
        for f in np.unique(fold):
            computed = metrics(y[fold == f], pred[key][fold == f])
            row = canonical[(canonical.model == name) & (canonical.fold == f)].iloc[0]
            for column, value in computed.items():
                if isinstance(value, int):
                    if value != int(row[column]):
                        mismatches.append([int(f), column, value, int(row[column])])
                else:
                    worst = max(worst, abs(value - float(row[column])))
        comparison[name] = {"count_mismatches": mismatches, "max_abs_metric_diff": worst}
    out["C_independent_vs_canonical"] = comparison

    per_fold_matlab = [
        metrics(y[fold == f], pred["pfs_matlab_line"][fold == f]) for f in np.unique(fold)
    ]
    per_fold_paper = [metrics(y[fold == f], pred["pfs"][fold == f]) for f in np.unique(fold)]
    keys = ("balanced_accuracy", "precision_positive", "recall_positive", "f1_positive", "mcc")
    out["C_pfs_paper_vs_matlab_line"] = {
        "rows_where_rules_disagree": int((pred["pfs"] != pred["pfs_matlab_line"]).sum()),
        "predicted_bankrupt_paper_rule": int(pred["pfs"].sum()),
        "predicted_bankrupt_matlab_line": int(pred["pfs_matlab_line"].sum()),
        "mean_paper_rule": {k: float(np.mean([m[k] for m in per_fold_paper])) for k in keys},
        "mean_matlab_line": {k: float(np.mean([m[k] for m in per_fold_matlab])) for k in keys},
    }

    # D. production classes, row-by-row
    sys.path.insert(0, str(ROOT / "src"))
    from taiwan_soft_classifiers.data import clean_data
    from taiwan_soft_classifiers.soft_classifiers import (
        FPFSKNNClassifier,
        IFPIFSHCClassifier,
        PFSKNNClassifier,
    )

    out["D_production_cleaning_equals_independent"] = bool(clean_data(raw)[0].equals(clean))
    factories = {
        "fpfs": lambda: FPFSKNNClassifier(k=3),
        "ifpifs": lambda: IFPIFSHCClassifier(lambda1=5, lambda2=0.5),
        "pfs": lambda: PFSKNNClassifier(k=3, lambda_value=0.5, p=5),
    }
    production = {k: np.empty(len(y), dtype=int) for k in factories}
    for f in np.unique(fold):
        test = fold == f
        for key, make in factories.items():
            production[key][test] = make().fit(x[~test], y[~test]).predict(x[test])
        print(f"production fold {f} done ({time.time() - start:.0f}s)", flush=True)
    out["D_prediction_mismatches_production_vs_independent"] = {
        NAMES[k]: int((production[k] != pred[k]).sum()) for k in factories
    }
    out["D_predictions_compared_per_model"] = len(y)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
