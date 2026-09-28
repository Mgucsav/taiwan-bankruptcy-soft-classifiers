"""Confusion-matrix counts and their consistency with the reported threshold metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.baselines import BaselineSpec, evaluate_baselines
from taiwan_soft_classifiers.evaluation import CONFUSION_COLUMNS, confusion_counts
from taiwan_soft_classifiers.splitting import make_fold_assignments

REPORTS = config.PROJECT_ROOT / "reports" / "pilot_results"
FOLD_FILES = ["baseline_fold_results.csv", "soft_classifier_fold_results.csv"]


def recompute_from_counts(df: pd.DataFrame) -> pd.DataFrame:
    tp, tn = df["true_positive"], df["true_negative"]
    fp, fn = df["false_positive"], df["false_negative"]
    precision = (tp / (tp + fp)).where(tp + fp > 0, 0.0)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return pd.DataFrame(
        {
            "precision_positive": precision,
            "recall_positive": recall,
            "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2,
        }
    )


def assert_counts_consistent(df: pd.DataFrame) -> None:
    counts = df[list(CONFUSION_COLUMNS)]
    assert (counts >= 0).all().all()
    assert (counts.sum(axis=1) == df["n_test"]).all()
    assert (df["true_positive"] + df["false_negative"] == df["positive_test"]).all()
    assert (df["true_positive"] + df["false_positive"] == df["predicted_positive"]).all()
    recomputed = recompute_from_counts(df)
    for metric in ("precision_positive", "recall_positive", "balanced_accuracy"):
        np.testing.assert_allclose(recomputed[metric], df[metric], rtol=0, atol=1e-12)


def test_confusion_counts_match_sklearn() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    counts = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    assert counts == {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }
    row = pd.DataFrame([{**counts}])
    recomputed = recompute_from_counts(row).iloc[0]
    assert recomputed["precision_positive"] == pytest.approx(precision_score(y_true, y_pred))
    assert recomputed["recall_positive"] == pytest.approx(recall_score(y_true, y_pred))
    assert recomputed["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert recomputed["specificity"] == pytest.approx(tn / (tn + fp))


def test_confusion_counts_without_predicted_positives() -> None:
    counts = confusion_counts(np.array([0, 1, 1, 0]), np.array([0, 0, 0, 0]))
    assert counts == {
        "true_positive": 0,
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 2,
    }


@pytest.mark.parametrize("name", FOLD_FILES)
def test_stored_fold_results_are_consistent_with_counts(name: str) -> None:
    df = pd.read_csv(Path(REPORTS) / name)
    assert list(df.columns[5:9]) == list(CONFUSION_COLUMNS)
    assert_counts_consistent(df)


def test_soft_results_expose_no_score_based_metrics() -> None:
    # The soft classifiers produce labels, not continuous scores: no ROC-AUC or PR-AUC.
    df = pd.read_csv(Path(REPORTS) / "soft_classifier_fold_results.csv")
    assert not {"roc_auc", "pr_auc"} & set(df.columns)


def test_baseline_evaluation_writes_consistent_counts(clean_df: pd.DataFrame) -> None:
    assignments = make_fold_assignments(clean_df[config.TARGET_COLUMN])
    specs = (
        BaselineSpec("Dummy-prior", lambda: DummyClassifier(strategy="prior")),
        BaselineSpec(
            "Logistic-balanced",
            lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=5_000,
                            random_state=config.RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    )
    results = evaluate_baselines(clean_df, assignments, specs)
    assert len(results) == 2 * config.N_SPLITS
    assert_counts_consistent(results)
    reference = pd.read_csv(Path(REPORTS) / "baseline_fold_results.csv")
    merged = results.merge(reference, on=["model", "fold"], suffixes=("", "_ref"))
    assert len(merged) == 2 * config.N_SPLITS
    for column in CONFUSION_COLUMNS:
        assert (merged[column] == merged[f"{column}_ref"]).all()
