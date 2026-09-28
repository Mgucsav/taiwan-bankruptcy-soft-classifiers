"""Fixed stratified 5-fold assignments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.splitting import (
    fold_class_counts,
    make_fold_assignments,
    validate_fold_assignments,
)
from taiwan_soft_classifiers.validation import DataValidationError


@pytest.fixture(scope="module")
def target(clean_df: pd.DataFrame) -> pd.Series:
    return clean_df[config.TARGET_COLUMN]


@pytest.fixture(scope="module")
def assignments(target: pd.Series) -> pd.DataFrame:
    return make_fold_assignments(target)


def test_assignments_are_valid(assignments: pd.DataFrame, target: pd.Series) -> None:
    validate_fold_assignments(assignments, target)


def test_every_row_in_exactly_one_fold(assignments: pd.DataFrame) -> None:
    assert list(assignments.columns) == ["row_id", "target", "fold"]
    assert len(assignments) == config.EXPECTED_CLEAN_ROWS
    assert assignments["row_id"].is_unique
    assert sorted(assignments["row_id"]) == list(range(config.EXPECTED_CLEAN_ROWS))
    assert sorted(assignments["fold"].unique()) == [0, 1, 2, 3, 4]


def test_assignments_are_deterministic(target: pd.Series, assignments: pd.DataFrame) -> None:
    again = make_fold_assignments(target, random_state=config.RANDOM_STATE)
    pd.testing.assert_frame_equal(assignments, again)


def test_different_seed_changes_assignments(target: pd.Series, assignments: pd.DataFrame) -> None:
    other = make_fold_assignments(target, random_state=config.RANDOM_STATE + 1)
    assert not np.array_equal(other["fold"].to_numpy(), assignments["fold"].to_numpy())


def test_fold_class_distribution(assignments: pd.DataFrame, target: pd.Series) -> None:
    counts = fold_class_counts(assignments)
    assert (counts["negative"] > 0).all()
    assert (counts["positive"] > 0).all()
    assert counts["negative"].sum() == 6599
    assert counts["positive"].sum() == 220
    assert counts["positive"].max() - counts["positive"].min() <= 1
    assert counts["negative"].max() - counts["negative"].min() <= 1
    overall = target.mean()
    assert (counts["positive_rate"] - overall).abs().max() <= config.FOLD_RATE_TOLERANCE


def test_saved_fold_file_matches_regeneration(assignments: pd.DataFrame) -> None:
    if not config.FOLD_ASSIGNMENTS_PATH.is_file():
        pytest.skip("Fold file missing; run `python scripts/prepare_data.py` first")
    saved = pd.read_csv(config.FOLD_ASSIGNMENTS_PATH)
    pd.testing.assert_frame_equal(saved, assignments, check_dtype=False)


def test_duplicate_row_id_is_rejected(assignments: pd.DataFrame, target: pd.Series) -> None:
    broken = assignments.copy()
    broken.loc[1, "row_id"] = 0
    with pytest.raises(DataValidationError, match="more than once"):
        validate_fold_assignments(broken, target)


def test_missing_fold_is_rejected(assignments: pd.DataFrame, target: pd.Series) -> None:
    broken = assignments.copy()
    broken.loc[broken["fold"] == 4, "fold"] = 3
    with pytest.raises(DataValidationError, match="Fold ids"):
        validate_fold_assignments(broken, target)


def test_single_class_fold_is_rejected(assignments: pd.DataFrame, target: pd.Series) -> None:
    broken = assignments.copy()
    broken.loc[(broken["fold"] == 0) & (broken["target"] == 1), "fold"] = 1
    with pytest.raises(DataValidationError, match="missing a class"):
        validate_fold_assignments(broken, target)
