"""Evaluate selected official-code soft classifiers on the fixed five folds."""

from __future__ import annotations

import json
import time

import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.data import load_clean_data, write_csv
from taiwan_soft_classifiers.soft_classifiers import (
    FPFSKNNClassifier,
    IFPIFSHCClassifier,
    PFSKNNClassifier,
)
from taiwan_soft_classifiers.splitting import validate_fold_assignments


def main() -> None:
    clean = load_clean_data()
    assignments = pd.read_csv(config.FOLD_ASSIGNMENTS_PATH)
    y = clean[config.TARGET_COLUMN].astype(int).reset_index(drop=True)
    x = clean.drop(columns=[config.TARGET_COLUMN]).reset_index(drop=True)
    validate_fold_assignments(assignments, y)

    factories = {
        "FPFS-kNN(k=3,Pearson)": lambda: FPFSKNNClassifier(k=3),
        "IFPIFS-HC(lambda1=5,lambda2=0.5)": lambda: IFPIFSHCClassifier(lambda1=5, lambda2=0.5),
        "PFS-kNN(k=3,lambda=0.5,p=5)": lambda: PFSKNNClassifier(k=3, lambda_value=0.5, p=5),
    }

    rows: list[dict[str, float | int | str]] = []
    for model_name, factory in factories.items():
        for fold_id in sorted(assignments["fold"].unique()):
            is_test = assignments["fold"].to_numpy() == fold_id
            model = factory()
            started = time.perf_counter()
            model.fit(x.loc[~is_test].to_numpy(), y.loc[~is_test].to_numpy())
            prediction = model.predict(x.loc[is_test].to_numpy())
            elapsed = time.perf_counter() - started
            y_test = y.loc[is_test].to_numpy()
            rows.append(
                {
                    "model": model_name,
                    "fold": int(fold_id),
                    "n_test": int(y_test.size),
                    "positive_test": int(y_test.sum()),
                    "predicted_positive": int(prediction.sum()),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
                    "precision_positive": float(
                        precision_score(y_test, prediction, pos_label=1, zero_division=0)
                    ),
                    "recall_positive": float(recall_score(y_test, prediction, pos_label=1)),
                    "f1_positive": float(f1_score(y_test, prediction, pos_label=1)),
                    "mcc": float(matthews_corrcoef(y_test, prediction)),
                    "elapsed_seconds": float(elapsed),
                }
            )
            print(model_name, "fold", fold_id, rows[-1])

    fold_results = pd.DataFrame(rows)
    metrics = [
        "balanced_accuracy",
        "precision_positive",
        "recall_positive",
        "f1_positive",
        "mcc",
        "elapsed_seconds",
    ]
    summary = fold_results.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    fold_path = config.ARTIFACTS_DIR / "soft_classifier_fold_results.csv"
    summary_path = config.ARTIFACTS_DIR / "soft_classifier_summary.csv"
    protocol_path = config.ARTIFACTS_DIR / "soft_classifier_protocol.json"
    write_csv(fold_results, fold_path)
    write_csv(summary, summary_path)
    protocol_path.write_text(
        json.dumps(
            {
                "status": "official_matlab_formula_port_pilot",
                "primary_protocol": "five fixed stratified folds",
                "normalization": (
                    "training-fold min-max with held-out clipping to [0,1]; intentional "
                    "leakage-control correction to public MATLAB demos"
                ),
                "hyperparameters": {
                    "FPFS-kNN": {"k": 3, "correlation": "Pearson", "metrics": 5},
                    "IFPIFS-HC": {"lambda1": 5, "lambda2": 0.5},
                    "PFS-kNN": {"k": 3, "lambda": 0.5, "p": 5},
                },
                "tuning": "none",
                "warning": (
                    "Python ports require numerical cross-check against MATLAB before claiming "
                    "bit-level implementation equivalence"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
