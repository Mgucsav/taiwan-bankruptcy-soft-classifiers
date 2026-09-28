"""Leakage-free, pre-specified classical baselines for the Taiwan study."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.evaluation import confusion_counts
from taiwan_soft_classifiers.splitting import validate_fold_assignments


@dataclass(frozen=True)
class BaselineSpec:
    """A named estimator factory; a fresh estimator is built for every fold."""

    name: str
    factory: Callable[[], ClassifierMixin]


def baseline_specs() -> tuple[BaselineSpec, ...]:
    """Return fixed, untuned baselines chosen before inspecting their performance."""

    return (
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
        BaselineSpec(
            "SVM-RBF-balanced",
            lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            class_weight="balanced",
                            probability=False,
                            random_state=config.RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        BaselineSpec(
            "RandomForest-balanced",
            lambda: RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    )


def _positive_score(estimator: ClassifierMixin, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_test)
        classes = np.asarray(estimator.classes_)
        positive_index = int(np.flatnonzero(classes == 1)[0])
        return np.asarray(probabilities[:, positive_index], dtype=float)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(x_test), dtype=float)
    raise TypeError(f"{type(estimator).__name__} exposes neither probability nor decision score")


def evaluate_baselines(
    clean: pd.DataFrame,
    assignments: pd.DataFrame,
    specs: tuple[BaselineSpec, ...] | None = None,
) -> pd.DataFrame:
    """Evaluate each baseline on the same stored five stratified test folds."""

    target_name = config.TARGET_COLUMN
    if target_name not in clean.columns:
        raise ValueError(f"Target column {target_name!r} is missing")

    y = clean[target_name].astype(int).reset_index(drop=True)
    x = clean.drop(columns=[target_name]).reset_index(drop=True)
    validate_fold_assignments(assignments, y)
    specs = specs or baseline_specs()

    rows: list[dict[str, float | int | str]] = []
    for spec in specs:
        for fold_id in sorted(assignments["fold"].unique()):
            is_test = assignments["fold"].to_numpy() == fold_id
            x_train, x_test = x.loc[~is_test], x.loc[is_test]
            y_train, y_test = y.loc[~is_test], y.loc[is_test]

            estimator = spec.factory()
            estimator.fit(x_train, y_train)
            prediction = np.asarray(estimator.predict(x_test), dtype=int)
            score = _positive_score(estimator, x_test)

            rows.append(
                {
                    "model": spec.name,
                    "fold": int(fold_id),
                    "n_test": int(y_test.shape[0]),
                    "positive_test": int(y_test.sum()),
                    "predicted_positive": int(prediction.sum()),
                    **confusion_counts(y_test.to_numpy(), prediction),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
                    "precision_positive": float(
                        precision_score(y_test, prediction, pos_label=1, zero_division=0)
                    ),
                    "recall_positive": float(recall_score(y_test, prediction, pos_label=1)),
                    "f1_positive": float(f1_score(y_test, prediction, pos_label=1)),
                    "mcc": float(matthews_corrcoef(y_test, prediction)),
                    "roc_auc": float(roc_auc_score(y_test, score)),
                    "pr_auc": float(average_precision_score(y_test, score)),
                }
            )
    return pd.DataFrame(rows)


def summarize_fold_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Return mean and sample standard deviation across the five held-out folds."""

    metrics = [
        "balanced_accuracy",
        "precision_positive",
        "recall_positive",
        "f1_positive",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]
    grouped = fold_results.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()
