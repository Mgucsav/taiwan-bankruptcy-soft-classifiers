"""Explicit data contract checks. Every failed check raises :class:`DataValidationError`."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from taiwan_soft_classifiers import config


class DataValidationError(ValueError):
    """Raised when a dataset violates its documented contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataValidationError(message)


def count_missing_cells(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def count_duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated(keep="first").sum())


def class_counts(y: pd.Series | np.ndarray) -> dict[int, int]:
    counts = pd.Series(np.asarray(y)).value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}


def validate_unique_names(names: Sequence[str], *, context: str) -> None:
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    _require(not duplicates, f"{context}: duplicate column names {duplicates}")


def validate_stripped_names(names: Iterable[str]) -> None:
    bad = [n for n in names if n != n.strip()]
    _require(not bad, f"Column names still contain leading/trailing whitespace: {bad}")


def validate_target(df: pd.DataFrame, expected_counts: dict[int, int]) -> None:
    target = config.TARGET_COLUMN
    _require(target in df.columns, f"Target column {target!r} not found")
    values = set(pd.unique(df[target]).tolist())
    _require(
        values <= set(config.EXPECTED_CLASSES),
        f"Target contains unexpected classes: {sorted(values - set(config.EXPECTED_CLASSES))}",
    )
    observed = class_counts(df[target])
    _require(
        observed == expected_counts,
        f"Class counts {observed} differ from expected {expected_counts}",
    )


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate the raw UCI CSV (after header whitespace has been stripped)."""
    _require(
        df.shape[0] == config.EXPECTED_RAW_ROWS,
        f"Raw data has {df.shape[0]} rows, expected {config.EXPECTED_RAW_ROWS}",
    )
    _require(
        df.shape[1] == config.EXPECTED_RAW_COLUMNS,
        f"Raw data has {df.shape[1]} columns, expected {config.EXPECTED_RAW_COLUMNS}",
    )
    validate_unique_names(list(df.columns), context="Raw data")
    validate_stripped_names(df.columns)
    n_features = df.shape[1] - 1 if config.TARGET_COLUMN in df.columns else df.shape[1]
    _require(
        n_features == config.EXPECTED_RAW_FEATURES,
        f"Raw data has {n_features} features, expected {config.EXPECTED_RAW_FEATURES}",
    )
    validate_target(df, config.EXPECTED_CLASS_COUNTS)
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    _require(not non_numeric, f"Non-numeric columns in raw data: {non_numeric}")
    missing = count_missing_cells(df)
    _require(
        missing == config.EXPECTED_MISSING_CELLS,
        f"Raw data has {missing} missing cells, expected {config.EXPECTED_MISSING_CELLS}",
    )
    duplicates = count_duplicate_rows(df)
    _require(
        duplicates == config.EXPECTED_DUPLICATE_ROWS,
        f"Raw data has {duplicates} duplicate rows, expected {config.EXPECTED_DUPLICATE_ROWS}",
    )


def find_constant_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns (target excluded) with a single unique value."""
    return [
        c for c in df.columns if c != config.TARGET_COLUMN and int(df[c].nunique(dropna=False)) == 1
    ]


def find_identical_column_pairs(
    df: pd.DataFrame, exclude: Iterable[str] = ()
) -> list[tuple[str, str]]:
    """Pairs ``(kept, duplicate)`` of feature columns whose values are exactly equal.

    The target and ``exclude`` columns are not compared. Columns are scanned in file order;
    each duplicate is paired with the first earlier column it equals, so the first column of
    every group of identical columns is kept.
    """
    skip = {config.TARGET_COLUMN, *exclude}
    features = [c for c in df.columns if c not in skip]
    values = {c: df[c].to_numpy() for c in features}
    pairs: list[tuple[str, str]] = []
    duplicates: set[str] = set()
    for j, later in enumerate(features):
        for earlier in features[:j]:
            if earlier not in duplicates and np.array_equal(values[earlier], values[later]):
                pairs.append((earlier, later))
                duplicates.add(later)
                break
    return pairs


def validate_constant_column(df: pd.DataFrame, column: str) -> None:
    _require(column in df.columns, f"Constant column {column!r} not found")
    n_unique = int(df[column].nunique(dropna=False))
    _require(n_unique == 1, f"Column {column!r} is not constant ({n_unique} unique values)")


def validate_identical_columns(df: pd.DataFrame, keep: str, drop: str) -> None:
    for column in (keep, drop):
        _require(column in df.columns, f"Column {column!r} not found")
    identical = np.array_equal(df[keep].to_numpy(), df[drop].to_numpy())
    _require(identical, f"Columns {keep!r} and {drop!r} are not exactly identical")


def validate_clean_data(df: pd.DataFrame) -> None:
    _require(
        df.shape == (config.EXPECTED_CLEAN_ROWS, config.EXPECTED_CLEAN_COLUMNS),
        f"Clean data shape {df.shape} differs from expected "
        f"{(config.EXPECTED_CLEAN_ROWS, config.EXPECTED_CLEAN_COLUMNS)}",
    )
    validate_unique_names(list(df.columns), context="Clean data")
    validate_stripped_names(df.columns)
    n_features = df.shape[1] - 1
    _require(
        n_features == config.EXPECTED_CLEAN_FEATURES,
        f"Clean data has {n_features} features, expected {config.EXPECTED_CLEAN_FEATURES}",
    )
    validate_target(df, config.EXPECTED_CLASS_COUNTS)
    missing = count_missing_cells(df)
    _require(missing == 0, f"Clean data has {missing} missing cells")
    constants = find_constant_columns(df)
    _require(not constants, f"Clean data still has constant columns: {constants}")
    pairs = find_identical_column_pairs(df)
    _require(not pairs, f"Clean data still has identical column pairs: {pairs}")
