"""Download the official UCI Taiwanese Bankruptcy Prediction ZIP and extract ``data.csv``.

Usage:
    python scripts/download_data.py [--force] [--expected-sha256 HEX]

An existing valid ZIP is reused (no network access) unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.data import (
    DataSourceError,
    download_file,
    extract_csv,
    load_download_metadata,
    sha256_file,
    utc_now_iso,
    validate_zip,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the UCI Taiwan bankruptcy data.")
    parser.add_argument(
        "--force", action="store_true", help="download again even if a valid ZIP exists"
    )
    parser.add_argument(
        "--expected-sha256", default=None, help="fail unless the ZIP has this SHA-256 digest"
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="network timeout in seconds")
    return parser.parse_args(argv)


def _existing_zip_is_valid(zip_path: Path) -> bool:
    if not zip_path.is_file():
        return False
    try:
        validate_zip(zip_path)
    except DataSourceError as exc:
        print(f"Existing ZIP is unusable ({exc}); downloading again.")
        return False
    return True


def _previous_download_time(zip_sha256: str) -> str | None:
    try:
        previous = load_download_metadata()
    except (DataSourceError, ValueError):
        return None
    if previous.get("zip_sha256") != zip_sha256:
        return None
    return previous.get("downloaded_at_utc")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    zip_path = config.RAW_ZIP_PATH
    csv_path = config.RAW_CSV_PATH

    try:
        if args.force or not _existing_zip_is_valid(zip_path):
            print(f"Downloading {config.UCI_ZIP_URL}")
            n_bytes = download_file(config.UCI_ZIP_URL, zip_path, timeout=args.timeout)
            downloaded_at: str | None = utc_now_iso()
            print(f"Saved {n_bytes} bytes to {zip_path.relative_to(config.PROJECT_ROOT)}")
            zip_sha256 = sha256_file(zip_path)
        else:
            print(
                f"Valid ZIP already present at {zip_path.relative_to(config.PROJECT_ROOT)}; "
                "skipping download (use --force to download again)."
            )
            zip_sha256 = sha256_file(zip_path)
            downloaded_at = _previous_download_time(zip_sha256)

        validate_zip(zip_path)
        if args.expected_sha256 and zip_sha256.lower() != args.expected_sha256.lower():
            raise DataSourceError(
                f"SHA-256 mismatch: expected {args.expected_sha256}, got {zip_sha256}"
            )

        member = extract_csv(zip_path, csv_path)
        metadata = {
            "source_url": config.UCI_ZIP_URL,
            "uci_page": config.UCI_PAGE_URL,
            "doi": config.DATASET_DOI,
            "license": config.DATASET_LICENSE,
            "downloaded_at_utc": downloaded_at,
            "zip_file": zip_path.name,
            "zip_size_bytes": zip_path.stat().st_size,
            "zip_sha256": zip_sha256,
            "csv_member": member.filename,
            "csv_file": csv_path.name,
            "csv_sha256": sha256_file(csv_path),
        }
        write_json(metadata, config.DOWNLOAD_METADATA_PATH)
    except DataSourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"ZIP SHA-256: {zip_sha256}")
    print(f"Extracted {member.filename!r} -> {csv_path.relative_to(config.PROJECT_ROOT)}")
    if downloaded_at is None:
        print("WARNING: original download time is unknown; rerun with --force to record it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
