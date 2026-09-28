"""Fixed stratified cross-validation folds shared by every classifier in the study."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.validation import DataValidationError


def make_fold_assignments(
    y: pd.Series | np.ndarray,
    *,
    n_splits: int = config.N_SPLITS,
    shuffle: bool = config.SHUFFLE,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Assign every row to exactly one test fold.

    ``row_id`` is the 0-based row position in the cleaned dataset (identical to the raw order).
    Only the target is used; no feature statistic is computed.
    """
    target = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    fold = np.full(target.shape[0], -1, dtype=int)
    for fold_id, (_, test_idx) in enumerate(skf.split(np.zeros((target.shape[0], 1)), target)):
        fold[test_idx] = fold_id
    return pd.DataFrame(
        {"row_id": np.arange(target.shape[0]), "target": target.astype(int), "fold": fold}
    )


def fold_class_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    """Negative/positive counts and bankruptcy rate per test fold."""
    counts = pd.crosstab(assignments["fold"], assignments["target"])
    counts = counts.reindex(columns=[0, 1], fill_value=0)
    counts.columns = ["negative", "positive"]
    counts["total"] = counts["negative"] + counts["positive"]
    counts["positive_rate"] = counts["positive"] / counts["total"]
    return counts


def validate_fold_assignments(
    assignments: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_splits: int = config.N_SPLITS,
    tolerance: float = config.FOLD_RATE_TOLERANCE,
) -> None:
    target = np.asarray(y).astype(int)
    n = target.shape[0]

    if list(assignments.columns) != ["row_id", "target", "fold"]:
        raise DataValidationError(f"Unexpected fold columns: {list(assignments.columns)}")
    if assignments.shape[0] != n:
        raise DataValidationError(f"Fold table has {assignments.shape[0]} rows, expected {n}")
    if assignments["row_id"].duplicated().any():
        raise DataValidationError("Some row_id values appear more than once")
    if set(assignments["row_id"]) != set(range(n)):
        raise DataValidationError("row_id values do not cover every row exactly once")
    ordered = assignments.sort_values("row_id")
    if not np.array_equal(ordered["target"].to_numpy(), target):
        raise DataValidationError("Fold table targets do not match the dataset target")
    if set(assignments["fold"]) != set(range(n_splits)):
        raise DataValidationError(
            f"Fold ids {sorted(set(assignments['fold']))} differ from {list(range(n_splits))}"
        )

    counts = fold_class_counts(assignments)
    if (counts[["negative", "positive"]] == 0).any().any():
        raise DataValidationError(f"A fold is missing a class:\n{counts}")
    overall_rate = float(target.mean())
    deviation = (counts["positive_rate"] - overall_rate).abs()
    if (deviation > tolerance).any():
        raise DataValidationError(
            f"Fold bankruptcy rates deviate from the overall rate {overall_rate:.5f} "
            f"by more than {tolerance}:\n{counts}"
        )
