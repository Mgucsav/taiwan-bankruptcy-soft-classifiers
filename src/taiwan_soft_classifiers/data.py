"""Download, safe extraction, loading, cleaning and profiling of the UCI Taiwan bankruptcy data.

No statistic computed here is fitted into any model: scaling, feature selection and class
balancing are deliberately absent and must be fitted exclusively on each training fold.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import IO, Any

import pandas as pd

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.validation import (
    DataValidationError,
    class_counts,
    count_duplicate_rows,
    count_missing_cells,
    find_constant_columns,
    find_identical_column_pairs,
    validate_unique_names,
)

_CHUNK_SIZE = 1 << 20


class DataSourceError(RuntimeError):
    """Raised when the data source cannot be downloaded, read or safely extracted."""


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@contextmanager
def atomic_write(path: Path, mode: str = "wb", **open_kwargs: Any) -> Iterator[IO[Any]]:
    """Write to a temporary file next to ``path`` and move it into place only on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, mode, **open_kwargs) as fh:
            yield fh
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(data: dict[str, Any], path: Path) -> None:
    with atomic_write(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def write_csv(df: pd.DataFrame, path: Path, **to_csv_kwargs: Any) -> None:
    with atomic_write(path, "w", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False, lineterminator="\n", **to_csv_kwargs)


# ---------------------------------------------------------------------------
# Download and extraction
# ---------------------------------------------------------------------------
def download_file(url: str, dest: Path, *, timeout: float = 60.0) -> int:
    """Download ``url`` to ``dest`` atomically; return the number of bytes written.

    A partially downloaded file never appears at ``dest``.
    """
    from taiwan_soft_classifiers import __version__

    request = urllib.request.Request(
        url, headers={"User-Agent": f"taiwan-soft-classifiers/{__version__}"}
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            atomic_write(dest, "wb") as fh,
        ):
            expected = response.headers.get("Content-Length")
            written = 0
            while chunk := response.read(_CHUNK_SIZE):
                fh.write(chunk)
                written += len(chunk)
            if expected is not None and int(expected) != written:
                raise DataSourceError(
                    f"Incomplete download from {url}: received {written} of {expected} bytes"
                )
    except urllib.error.HTTPError as exc:
        raise DataSourceError(
            f"HTTP error {exc.code} ({exc.reason}) while downloading {url}"
        ) from exc
    except urllib.error.URLError as exc:
        raise DataSourceError(f"Network error while downloading {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise DataSourceError(f"Timed out after {timeout} s while downloading {url}") from exc
    return written


def _is_safe_member_name(name: str) -> bool:
    if "\\" in name or ":" in name:
        return False
    parts = PurePosixPath(name)
    return not parts.is_absolute() and ".." not in parts.parts


def find_csv_member(zf: zipfile.ZipFile, member_name: str = config.RAW_CSV_NAME) -> zipfile.ZipInfo:
    """Return the single archive member whose basename is ``member_name``."""
    candidates = [
        info
        for info in zf.infolist()
        if not info.is_dir() and PurePosixPath(info.filename.replace("\\", "/")).name == member_name
    ]
    if len(candidates) != 1:
        names = [info.filename for info in zf.infolist()]
        raise DataSourceError(
            f"Expected exactly one {member_name!r} in the archive, found {len(candidates)}; "
            f"archive members: {names}"
        )
    info = candidates[0]
    if not _is_safe_member_name(info.filename):
        raise DataSourceError(f"Unsafe path in archive member: {info.filename!r}")
    if info.file_size > config.MAX_CSV_BYTES:
        raise DataSourceError(
            f"Archive member {info.filename!r} is {info.file_size} bytes, "
            f"above the {config.MAX_CSV_BYTES}-byte safety limit"
        )
    return info


def validate_zip(zip_path: Path, member_name: str = config.RAW_CSV_NAME) -> zipfile.ZipInfo:
    """Check that ``zip_path`` is a readable ZIP containing the expected member."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return find_csv_member(zf, member_name)
    except zipfile.BadZipFile as exc:
        raise DataSourceError(f"{zip_path} is not a valid ZIP archive: {exc}") from exc


def extract_csv(
    zip_path: Path, dest: Path, member_name: str = config.RAW_CSV_NAME
) -> zipfile.ZipInfo:
    """Extract only ``member_name`` from ``zip_path`` to ``dest`` (no ``extractall``).

    The member is streamed with a size limit; ZIP CRC checks are enforced while reading.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            info = find_csv_member(zf, member_name)
            with zf.open(info) as src, atomic_write(dest, "wb") as out:
                written = 0
                while chunk := src.read(_CHUNK_SIZE):
                    written += len(chunk)
                    if written > config.MAX_CSV_BYTES:
                        raise DataSourceError(
                            f"Archive member {info.filename!r} exceeds the safety limit"
                        )
                    out.write(chunk)
    except zipfile.BadZipFile as exc:
        raise DataSourceError(f"Could not read {zip_path}: {exc}") from exc
    return info


def load_download_metadata(path: Path = config.DOWNLOAD_METADATA_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise DataSourceError(
            f"Download metadata not found at {path}; run `python scripts/download_data.py` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------
def strip_column_names(columns: list[str]) -> list[str]:
    return [str(c).strip() for c in columns]


def read_csv_header(path: Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def load_raw_data(path: Path = config.RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw CSV and strip whitespace around column names.

    Names are checked for uniqueness both before and after stripping, using the literal CSV
    header so that pandas' silent renaming of duplicate names cannot hide a collision.
    """
    path = Path(path)
    if not path.is_file():
        raise DataSourceError(
            f"Raw CSV not found at {path}; run `python scripts/download_data.py` first"
        )
    header = read_csv_header(path)
    validate_unique_names(header, context="Raw CSV header")
    stripped = strip_column_names(header)
    validate_unique_names(stripped, context="Raw CSV header after stripping whitespace")
    df = pd.read_csv(path, encoding="utf-8", float_precision="round_trip")
    if list(df.columns) != header:
        raise DataValidationError("pandas altered the raw CSV header while parsing")
    df.columns = stripped
    return df


def find_whitespace_names(header: list[str]) -> list[str]:
    """Raw header names that carry leading or trailing whitespace (reported, then stripped)."""
    return [name for name in header if name != name.strip()]


class CleaningStructureError(DataValidationError):
    """The data do not show the expected constant/duplicate structure; nothing is dropped."""

    def __init__(self, message: str, findings: dict[str, Any]) -> None:
        super().__init__(message)
        self.findings = findings


def _single_value(s: pd.Series) -> Any:
    value = s.to_numpy()[0]
    return value.item() if hasattr(value, "item") else value


def analyze_redundant_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Detect constant features and exactly identical feature pairs from the values alone."""
    constants = find_constant_columns(df)
    pairs = find_identical_column_pairs(df, exclude=constants)
    dropped = [
        {"column": c, "reason": f"constant: {_single_value(df[c])!r} in all {len(df)} rows"}
        for c in constants
    ] + [
        {
            "column": dup,
            "reason": f"exact duplicate: all {len(df)} values numerically equal to {keep!r}",
        }
        for keep, dup in pairs
    ]
    n_rows, n_cols = df.shape
    return {
        "constant_columns": constants,
        "duplicate_column_pairs": [{"kept": keep, "dropped": dup} for keep, dup in pairs],
        "proposed_drops": dropped,
        "shape_before": [n_rows, n_cols],
        "shape_after": [n_rows, n_cols - len(dropped)],
    }


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop constant features and later copies of exactly identical features.

    Columns are chosen by numerical checks on the data, never by name. If the detected
    structure differs from the expected one (see ``config``), :class:`CleaningStructureError`
    is raised with the full findings and no column is dropped.
    """
    df = raw.copy()
    df.columns = strip_column_names(list(df.columns))
    validate_unique_names(list(df.columns), context="Data after stripping column names")

    findings = analyze_redundant_columns(df)
    drops = [d["column"] for d in findings["proposed_drops"]]
    problems = []
    if findings["constant_columns"] != list(config.EXPECTED_CONSTANT_COLUMNS):
        problems.append(
            f"constant columns {findings['constant_columns']} differ from the expected "
            f"{list(config.EXPECTED_CONSTANT_COLUMNS)}"
        )
    if len(findings["duplicate_column_pairs"]) != config.EXPECTED_DUPLICATE_DROP_COUNT:
        problems.append(
            f"{len(findings['duplicate_column_pairs'])} duplicate columns found, expected "
            f"{config.EXPECTED_DUPLICATE_DROP_COUNT}"
        )
    protected = sorted(set(drops) & set(config.PROTECTED_COLUMNS))
    if protected:
        problems.append(f"protected columns would be dropped: {protected}")
    if problems:
        raise CleaningStructureError("; ".join(problems), findings)

    return df.drop(columns=drops), findings


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != config.TARGET_COLUMN]


def load_clean_data(path: Path = config.CLEAN_CSV_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DataSourceError(
            f"Clean CSV not found at {path}; run `python scripts/prepare_data.py` first"
        )
    return pd.read_csv(path, encoding="utf-8", float_precision="round_trip")


# ---------------------------------------------------------------------------
# Descriptive artefacts (never fed into model fitting)
# ---------------------------------------------------------------------------
def build_data_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in feature_columns(df):
        s = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique(dropna=True)),
                "min": s.min(),
                "q01": s.quantile(0.01),
                "median": s.median(),
                "mean": s.mean(),
                "std": s.std(ddof=1),
                "q99": s.quantile(0.99),
                "max": s.max(),
                "zero_ratio": float((s == 0).mean()),
                "missing_ratio": float(s.isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def _numeric_extreme(df: pd.DataFrame, columns: list[str], *, maximum: bool) -> dict[str, Any]:
    values = df[columns].max() if maximum else df[columns].min()
    column = values.idxmax() if maximum else values.idxmin()
    return {"value": float(values[column]), "column": str(column)}


def build_data_quality_report(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    cleaning_report: dict[str, Any],
    download_metadata: dict[str, Any],
    whitespace_columns: list[str],
) -> dict[str, Any]:
    target = config.TARGET_COLUMN
    counts = class_counts(clean[target])
    n = int(clean.shape[0])
    raw_features = feature_columns(raw)
    return {
        "dataset": config.DATASET_NAME,
        "source": {
            "uci_page": config.UCI_PAGE_URL,
            "zip_url": download_metadata.get("source_url", config.UCI_ZIP_URL),
            "doi": config.DATASET_DOI,
            "license": config.DATASET_LICENSE,
            "license_url": config.DATASET_LICENSE_URL,
        },
        "download": {
            "downloaded_at_utc": download_metadata.get("downloaded_at_utc"),
            "zip_sha256": download_metadata.get("zip_sha256"),
            "zip_size_bytes": download_metadata.get("zip_size_bytes"),
            "csv_member": download_metadata.get("csv_member"),
            "csv_sha256": download_metadata.get("csv_sha256"),
        },
        "target_column": target,
        "raw_shape": {"rows": int(raw.shape[0]), "columns": int(raw.shape[1])},
        "raw_feature_count": len(raw_features),
        "clean_shape": {"rows": n, "columns": int(clean.shape[1])},
        "clean_feature_count": len(feature_columns(clean)),
        "class_counts": {str(k): v for k, v in counts.items()},
        "class_ratios": {str(k): v / n for k, v in counts.items()},
        "missing_cells": {"raw": count_missing_cells(raw), "clean": count_missing_cells(clean)},
        "duplicate_rows": {"raw": count_duplicate_rows(raw), "clean": count_duplicate_rows(clean)},
        "column_name_whitespace": {
            "count": len(whitespace_columns),
            "action": "leading/trailing whitespace stripped; names verified unique afterwards",
            "raw_names": whitespace_columns,
        },
        "constant_columns": cleaning_report["constant_columns"],
        "duplicate_column_pairs": cleaning_report["duplicate_column_pairs"],
        "dropped_columns": cleaning_report["proposed_drops"],
        "raw_numeric_range": {
            "scope": "all raw explanatory features",
            "min": _numeric_extreme(raw, raw_features, maximum=False),
            "max": _numeric_extreme(raw, raw_features, maximum=True),
        },
        "raw_target_dtype": str(raw[target].dtype),
    }
