"""Loading, cleaning, safe extraction and Git hygiene."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.data import (
    CleaningStructureError,
    DataSourceError,
    build_data_profile,
    clean_data,
    extract_csv,
    find_whitespace_names,
    load_clean_data,
    load_download_metadata,
    read_csv_header,
    sha256_file,
    strip_column_names,
)


def test_column_names_are_stripped_and_unique(raw_df: pd.DataFrame) -> None:
    header = read_csv_header(config.RAW_CSV_PATH)
    assert any(name != name.strip() for name in header), "raw header should need stripping"
    assert list(raw_df.columns) == strip_column_names(header)
    assert all(name == name.strip() for name in raw_df.columns)
    assert raw_df.columns.is_unique


def test_clean_shape_and_columns(raw_df: pd.DataFrame) -> None:
    clean, report = clean_data(raw_df)
    assert clean.shape == (config.EXPECTED_CLEAN_ROWS, config.EXPECTED_CLEAN_COLUMNS)
    assert clean.shape[1] - 1 == config.EXPECTED_CLEAN_FEATURES
    assert report["constant_columns"] == ["Net Income Flag"]
    assert report["duplicate_column_pairs"] == [
        {"kept": "Current Liabilities/Liability", "dropped": "Current Liability to Liability"},
        {"kept": "Current Liabilities/Equity", "dropped": "Current Liability to Equity"},
    ]
    assert [d["column"] for d in report["proposed_drops"]] == [
        "Net Income Flag",
        "Current Liability to Liability",
        "Current Liability to Equity",
    ]
    assert all(d["reason"] for d in report["proposed_drops"])
    for pair in report["duplicate_column_pairs"]:
        assert pair["kept"] in clean.columns
        assert pair["dropped"] not in clean.columns
    assert "Net Income Flag" not in clean.columns
    assert "Liability-Assets Flag" in clean.columns
    assert config.TARGET_COLUMN in clean.columns
    assert not any(c.endswith((".1", ".2")) for c in clean.columns)


def test_unexpected_structure_stops_without_dropping(raw_df: pd.DataFrame) -> None:
    extra = raw_df.copy()
    first_feature = next(c for c in extra.columns if c != config.TARGET_COLUMN)
    extra[f"{first_feature} copy"] = extra[first_feature]
    with pytest.raises(CleaningStructureError, match="3 duplicate columns") as info:
        clean_data(extra)
    assert len(info.value.findings["duplicate_column_pairs"]) == 3
    assert info.value.findings["shape_before"] == [6819, 97]


def test_protected_column_is_never_dropped_silently(raw_df: pd.DataFrame) -> None:
    broken = raw_df.copy()
    broken["Liability-Assets Flag"] = 0
    with pytest.raises(CleaningStructureError, match="Liability-Assets Flag"):
        clean_data(broken)


def test_whitespace_names_are_reported(raw_df: pd.DataFrame) -> None:
    header = read_csv_header(config.RAW_CSV_PATH)
    reported = find_whitespace_names(header)
    assert reported == [name for name in header if name != name.strip()]
    assert len(reported) > 0


def test_cleaning_preserves_rows_classes_and_values(raw_df: pd.DataFrame) -> None:
    clean, _ = clean_data(raw_df)
    target = config.TARGET_COLUMN
    assert clean[target].value_counts().to_dict() == raw_df[target].value_counts().to_dict()
    assert int(clean.isna().sum().sum()) == 0
    pd.testing.assert_frame_equal(clean, raw_df[clean.columns])


def test_cleaning_does_not_modify_input(raw_df: pd.DataFrame) -> None:
    before = raw_df.copy()
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_clean_csv_round_trip(clean_df: pd.DataFrame) -> None:
    if not config.CLEAN_CSV_PATH.is_file():
        pytest.skip("Clean data missing; run `python scripts/prepare_data.py` first")
    written = load_clean_data(config.CLEAN_CSV_PATH)
    pd.testing.assert_frame_equal(written, clean_df, check_exact=True)


def test_profile_covers_every_clean_feature(clean_df: pd.DataFrame) -> None:
    profile = build_data_profile(clean_df)
    assert list(profile["column"]) == [c for c in clean_df.columns if c != config.TARGET_COLUMN]
    assert list(profile.columns) == [
        "column", "dtype", "n_unique", "min", "q01", "median", "mean",
        "std", "q99", "max", "zero_ratio", "missing_ratio",
    ]  # fmt: skip
    assert (profile["missing_ratio"] == 0).all()
    assert (profile["min"] <= profile["q01"]).all()
    assert (profile["q99"] <= profile["max"]).all()


def test_recorded_checksums_match_local_files() -> None:
    if not config.RAW_ZIP_PATH.is_file():
        pytest.skip("Raw ZIP missing; run `python scripts/download_data.py` first")
    metadata = load_download_metadata()
    assert sha256_file(config.RAW_ZIP_PATH) == metadata["zip_sha256"]
    assert sha256_file(config.RAW_CSV_PATH) == metadata["csv_sha256"]


def test_extract_csv_matches_archive_member(tmp_path: Path) -> None:
    if not config.RAW_ZIP_PATH.is_file():
        pytest.skip("Raw ZIP missing; run `python scripts/download_data.py` first")
    dest = tmp_path / "data.csv"
    extract_csv(config.RAW_ZIP_PATH, dest)
    assert sha256_file(dest) == sha256_file(config.RAW_CSV_PATH)
    assert [p.name for p in tmp_path.iterdir()] == ["data.csv"], "no temp files left behind"


def test_extract_rejects_path_traversal(tmp_path: Path) -> None:
    if not config.RAW_CSV_PATH.is_file():
        pytest.skip("Raw data missing; run `python scripts/download_data.py` first")
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(config.RAW_CSV_PATH, arcname="../data.csv")
    with pytest.raises(DataSourceError, match="Unsafe path"):
        extract_csv(archive, tmp_path / "out" / "data.csv")
    assert not (tmp_path / "out" / "data.csv").exists()


def test_extract_rejects_corrupt_archive(tmp_path: Path) -> None:
    archive = tmp_path / "corrupt.zip"
    archive.write_bytes(b"this is not a zip archive")
    with pytest.raises(DataSourceError, match="Could not read"):
        extract_csv(archive, tmp_path / "data.csv")
    assert not (tmp_path / "data.csv").exists()


# ---------------------------------------------------------------------------
# Git hygiene
# ---------------------------------------------------------------------------
def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def git_available() -> None:
    if shutil.which("git") is None or _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("git repository not available")


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/raw/taiwanese_bankruptcy_prediction.zip",
        "data/raw/data.csv",
        "data/raw/download_metadata.json",
        "data/processed/taiwan_bankruptcy_clean.csv",
    ],
)
def test_data_files_are_ignored(git_available: None, relative_path: str) -> None:
    assert _git("check-ignore", "-q", relative_path).returncode == 0


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "artifacts/data_quality.json",
        "artifacts/data_profile.csv",
        "artifacts/fold_assignments.csv",
    ],
)
def test_keep_files_and_artifacts_are_not_ignored(git_available: None, relative_path: str) -> None:
    assert _git("check-ignore", "-q", relative_path).returncode == 1


def test_no_data_files_are_tracked(git_available: None) -> None:
    tracked = _git("ls-files", "data").stdout.split()
    assert set(tracked) <= {"data/README.md", "data/raw/.gitkeep", "data/processed/.gitkeep"}
