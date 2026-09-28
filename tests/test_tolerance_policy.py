"""The two-tier reproducibility policy: exact / 1e-12 everywhere, atol 1e-3 only for the
Logistic-balanced ROC-AUC and PR-AUC (and their mean/std)."""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_soft_classifiers.evaluation import CONFUSION_COLUMNS
from taiwan_soft_classifiers.reproducibility import (
    CANONICAL_DIR,
    SCORE_ATOL,
    SCORE_METRICS,
    compare_tables,
    read_table,
    tolerance_for,
)

BASE_FOLDS = "baseline_fold_results.csv"
BASE_SUMMARY = "baseline_summary.csv"
SOFT_FOLDS = "soft_classifier_fold_results.csv"
SOFT_SUMMARY = "soft_classifier_summary.csv"
LOGISTIC = "Logistic-balanced"


def table(name: str) -> pd.DataFrame:
    return read_table(CANONICAL_DIR / name)


def perturb(df: pd.DataFrame, model: str, column: str, delta: float, fold: int | None = None):
    changed = df.copy()
    mask = changed["model"] == model
    if fold is not None:
        mask &= changed["fold"] == fold
    changed.loc[mask, column] = changed.loc[mask, column] + delta
    return changed


def test_policy_constants() -> None:
    assert SCORE_ATOL == 1e-3
    assert tolerance_for(LOGISTIC, "roc_auc") == (0.0, 1e-3)
    assert tolerance_for(LOGISTIC, "pr_auc_std") == (0.0, 1e-3)
    assert tolerance_for(LOGISTIC, "balanced_accuracy") == (1e-12, 1e-12)
    assert tolerance_for("SVM-RBF-balanced", "roc_auc") == (1e-12, 1e-12)
    assert tolerance_for("FPFS-kNN(k=3,Pearson)", "mcc") == (1e-12, 1e-12)


@pytest.mark.parametrize("column", ["roc_auc", "pr_auc"])
def test_logistic_fold_auc_difference_below_1e_3_passes(column: str) -> None:
    df = table(BASE_FOLDS)
    assert compare_tables(df, perturb(df, LOGISTIC, column, 9e-4), BASE_FOLDS) == []
    assert compare_tables(df, perturb(df, LOGISTIC, column, -9e-4), BASE_FOLDS) == []


@pytest.mark.parametrize("column", sorted(SCORE_METRICS - {"roc_auc", "pr_auc"}))
def test_logistic_summary_auc_difference_below_1e_3_passes(column: str) -> None:
    df = table(BASE_SUMMARY)
    assert compare_tables(df, perturb(df, LOGISTIC, column, 9e-4), BASE_SUMMARY) == []


@pytest.mark.parametrize("column", ["roc_auc", "pr_auc"])
def test_logistic_fold_auc_difference_above_1e_3_fails(column: str) -> None:
    df = table(BASE_FOLDS)
    differences = compare_tables(df, perturb(df, LOGISTIC, column, 1.1e-3, fold=2), BASE_FOLDS)
    assert [(d.location, d.column) for d in differences] == [(f"model={LOGISTIC}, fold=2", column)]
    assert differences[0].tolerance == "rtol=0, atol=0.001"


@pytest.mark.parametrize("column", sorted(SCORE_METRICS - {"roc_auc", "pr_auc"}))
def test_logistic_summary_auc_difference_above_1e_3_fails(column: str) -> None:
    df = table(BASE_SUMMARY)
    differences = compare_tables(df, perturb(df, LOGISTIC, column, 1.1e-3), BASE_SUMMARY)
    assert [d.column for d in differences] == [column]


@pytest.mark.parametrize("model", ["Dummy-prior", "SVM-RBF-balanced", "RandomForest-balanced"])
@pytest.mark.parametrize("column", ["roc_auc", "pr_auc"])
def test_other_models_auc_difference_of_1e_6_fails(model: str, column: str) -> None:
    df = table(BASE_FOLDS)
    differences = compare_tables(df, perturb(df, model, column, 1e-6, fold=0), BASE_FOLDS)
    assert [(d.location, d.column) for d in differences] == [(f"model={model}, fold=0", column)]
    summary = table(BASE_SUMMARY)
    assert compare_tables(summary, perturb(summary, model, f"{column}_mean", 1e-6), BASE_SUMMARY)


@pytest.mark.parametrize("column", ["balanced_accuracy", "precision_positive", "mcc"])
def test_logistic_threshold_metrics_stay_strict(column: str) -> None:
    df = table(BASE_FOLDS)
    assert compare_tables(df, perturb(df, LOGISTIC, column, 1e-9, fold=1), BASE_FOLDS)


@pytest.mark.parametrize("name", [BASE_FOLDS, SOFT_FOLDS])
@pytest.mark.parametrize("column", CONFUSION_COLUMNS)
def test_single_confusion_count_change_fails(name: str, column: str) -> None:
    df = table(name)
    changed = df.copy()
    changed.loc[4, column] += 1
    differences = compare_tables(df, changed, name)
    assert [d.column for d in differences] == [column]
    assert differences[0].abs_diff == 1.0


@pytest.mark.parametrize("name", [SOFT_FOLDS, SOFT_SUMMARY])
def test_any_soft_metric_difference_of_1e_9_fails(name: str) -> None:
    df = table(name)
    float_columns = [
        c
        for c in df.columns
        if pd.api.types.is_float_dtype(df[c]) and not c.startswith("elapsed_seconds")
    ]
    assert float_columns
    for column in float_columns:
        changed = df.copy()
        changed.loc[0, column] += 1e-9
        differences = compare_tables(df, changed, name)
        assert [d.column for d in differences] == [column], column


@pytest.mark.parametrize("name", [BASE_FOLDS, SOFT_FOLDS])
def test_model_name_or_fold_change_fails(name: str) -> None:
    df = table(name)
    renamed = df.copy()
    renamed.loc[0, "model"] = "renamed-model"
    assert compare_tables(df, renamed, name)
    refolded = df.copy()
    refolded.loc[0, "fold"] = 7
    assert compare_tables(df, refolded, name)


def test_cross_runner_values_observed_in_run_36431187367_pass() -> None:
    # Values produced on a different runner (Azure westcentralus) with identical package
    # versions; the strict 1e-12 gate rejected them, the declared policy accepts them.
    folds = table(BASE_FOLDS)
    observed = folds.copy()
    logistic = observed["model"] == LOGISTIC
    observed.loc[logistic & (observed["fold"] == 3), "roc_auc"] = 0.8434573002754822
    observed.loc[logistic & (observed["fold"] == 3), "pr_auc"] = 0.28182766952800964
    observed.loc[logistic & (observed["fold"] == 4), "pr_auc"] = 0.4564012096397687
    assert compare_tables(folds, observed, BASE_FOLDS) == []

    summary = table(BASE_SUMMARY)
    observed_summary = summary.copy()
    row = observed_summary["model"] == LOGISTIC
    observed_summary.loc[row, "roc_auc_mean"] = 0.8931648655380047
    observed_summary.loc[row, "roc_auc_std"] = 0.03689292364643384
    observed_summary.loc[row, "pr_auc_mean"] = 0.352753971831069
    observed_summary.loc[row, "pr_auc_std"] = 0.07434707092357354
    assert compare_tables(summary, observed_summary, BASE_SUMMARY) == []
