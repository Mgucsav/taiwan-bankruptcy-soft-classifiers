"""Raw and clean data contracts, checked on the real UCI data.

Failure cases are produced by perturbing copies of the real data, not by synthetic data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.validation import (
    DataValidationError,
    class_counts,
    count_duplicate_rows,
    count_missing_cells,
    find_constant_columns,
    find_identical_column_pairs,
    validate_clean_data,
    validate_constant_column,
    validate_identical_columns,
    validate_raw_data,
    validate_unique_names,
)

# Independent expectation for the real UCI file (the code detects pairs from values only).
KNOWN_DUPLICATE_PAIRS = (
    ("Current Liabilities/Liability", "Current Liability to Liability"),
    ("Current Liabilities/Equity", "Current Liability to Equity"),
)


def test_raw_contract_passes(raw_df: pd.DataFrame) -> None:
    validate_raw_data(raw_df)
    assert raw_df.shape == (config.EXPECTED_RAW_ROWS, config.EXPECTED_RAW_COLUMNS)
    assert config.TARGET_COLUMN in raw_df.columns
    assert raw_df.shape[1] - 1 == config.EXPECTED_RAW_FEATURES


def test_target_class_counts(raw_df: pd.DataFrame) -> None:
    target = raw_df[config.TARGET_COLUMN]
    assert set(target.unique()) == {0, 1}
    assert class_counts(target) == {0: 6599, 1: 220}


def test_no_missing_cells_or_duplicate_rows(raw_df: pd.DataFrame) -> None:
    assert count_missing_cells(raw_df) == 0
    assert count_duplicate_rows(raw_df) == 0


def test_missing_cell_is_rejected(raw_df: pd.DataFrame) -> None:
    broken = raw_df.copy()
    broken.iloc[0, 1] = float("nan")
    with pytest.raises(DataValidationError, match="missing"):
        validate_raw_data(broken)


def test_duplicate_row_is_rejected(raw_df: pd.DataFrame) -> None:
    broken = raw_df.copy()
    first, second = broken.index[broken[config.TARGET_COLUMN] == 0][:2]
    broken.loc[second] = broken.loc[first]
    with pytest.raises(DataValidationError, match="duplicate rows"):
        validate_raw_data(broken)


def test_wrong_row_count_is_rejected(raw_df: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="rows"):
        validate_raw_data(raw_df.iloc[:-1])


def test_unexpected_class_is_rejected(raw_df: pd.DataFrame) -> None:
    broken = raw_df.copy()
    broken.loc[0, config.TARGET_COLUMN] = 2
    with pytest.raises(DataValidationError, match="unexpected classes"):
        validate_raw_data(broken)


def test_class_count_change_is_rejected(raw_df: pd.DataFrame) -> None:
    broken = raw_df.copy()
    first_negative = broken.index[broken[config.TARGET_COLUMN] == 0][0]
    broken.loc[first_negative, config.TARGET_COLUMN] = 1
    with pytest.raises(DataValidationError, match="Class counts"):
        validate_raw_data(broken)


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="duplicate column names"):
        validate_unique_names(["ROA(C)", "ROA(C)"], context="test")


def test_constant_column_check(raw_df: pd.DataFrame) -> None:
    validate_constant_column(raw_df, "Net Income Flag")
    broken = raw_df.copy()
    broken.loc[0, "Net Income Flag"] = broken.loc[0, "Net Income Flag"] + 1
    with pytest.raises(DataValidationError, match="not constant"):
        validate_constant_column(broken, "Net Income Flag")


def test_detected_constant_columns(raw_df: pd.DataFrame) -> None:
    assert find_constant_columns(raw_df) == ["Net Income Flag"]


def test_detected_identical_pairs(raw_df: pd.DataFrame) -> None:
    pairs = find_identical_column_pairs(raw_df, exclude=find_constant_columns(raw_df))
    assert pairs == list(KNOWN_DUPLICATE_PAIRS)


def test_liability_assets_flag_is_neither_constant_nor_duplicate(raw_df: pd.DataFrame) -> None:
    column = "Liability-Assets Flag"
    assert column not in find_constant_columns(raw_df)
    assert all(column not in pair for pair in find_identical_column_pairs(raw_df))


def test_target_is_not_compared_as_feature(raw_df: pd.DataFrame) -> None:
    probe = raw_df.copy()
    probe["target copy"] = probe[config.TARGET_COLUMN]
    pairs = find_identical_column_pairs(probe)
    assert all(config.TARGET_COLUMN not in pair for pair in pairs)


@pytest.mark.parametrize(("keep", "drop"), KNOWN_DUPLICATE_PAIRS)
def test_duplicate_columns_are_identical(raw_df: pd.DataFrame, keep: str, drop: str) -> None:
    validate_identical_columns(raw_df, keep, drop)
    assert raw_df[keep].equals(raw_df[drop])


@pytest.mark.parametrize(("keep", "drop"), KNOWN_DUPLICATE_PAIRS)
def test_non_identical_columns_are_rejected(raw_df: pd.DataFrame, keep: str, drop: str) -> None:
    broken = raw_df.copy()
    broken.loc[0, drop] = broken.loc[0, drop] + 1.0
    with pytest.raises(DataValidationError, match="not exactly identical"):
        validate_identical_columns(broken, keep, drop)


def test_clean_contract_passes(clean_df: pd.DataFrame) -> None:
    validate_clean_data(clean_df)


def test_clean_contract_rejects_leftover_columns(raw_df: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        validate_clean_data(raw_df)
