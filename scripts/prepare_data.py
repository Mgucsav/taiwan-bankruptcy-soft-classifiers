"""Validate and clean the raw CSV, then write the clean CSV and reproducible artefacts.

Usage:
    python scripts/prepare_data.py

Outputs:
    data/processed/taiwan_bankruptcy_clean.csv   (not tracked by Git)
    artifacts/data_quality.json
    artifacts/data_profile.csv
    artifacts/fold_assignments.csv

No scaling, feature selection or resampling is performed here: those steps are fitted
exclusively on each training fold in later stages.
"""

from __future__ import annotations

import json
import sys

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.data import (
    CleaningStructureError,
    DataSourceError,
    build_data_profile,
    build_data_quality_report,
    clean_data,
    find_whitespace_names,
    load_download_metadata,
    load_raw_data,
    read_csv_header,
    sha256_file,
    write_csv,
    write_json,
)
from taiwan_soft_classifiers.splitting import (
    fold_class_counts,
    make_fold_assignments,
    validate_fold_assignments,
)
from taiwan_soft_classifiers.validation import (
    DataValidationError,
    validate_clean_data,
    validate_raw_data,
)


def main() -> int:
    try:
        metadata = load_download_metadata()
        csv_path = config.RAW_CSV_PATH
        if not csv_path.is_file() or sha256_file(csv_path) != metadata.get("csv_sha256"):
            raise DataSourceError(
                "data/raw/data.csv does not match the recorded download; "
                "run `python scripts/download_data.py` again"
            )

        whitespace_columns = find_whitespace_names(read_csv_header(csv_path))
        raw = load_raw_data(csv_path)
        validate_raw_data(raw)
        clean, cleaning_report = clean_data(raw)
        validate_clean_data(clean)

        assignments = make_fold_assignments(clean[config.TARGET_COLUMN])
        validate_fold_assignments(assignments, clean[config.TARGET_COLUMN])
    except CleaningStructureError as exc:
        print(f"ERROR: unexpected column structure, nothing dropped: {exc}", file=sys.stderr)
        print(json.dumps(exc.findings, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    except (DataSourceError, DataValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    write_csv(clean, config.CLEAN_CSV_PATH)
    write_json(
        build_data_quality_report(raw, clean, cleaning_report, metadata, whitespace_columns),
        config.DATA_QUALITY_PATH,
    )
    write_csv(build_data_profile(clean), config.DATA_PROFILE_PATH, float_format="%.12g")
    write_csv(assignments, config.FOLD_ASSIGNMENTS_PATH)

    print(f"Raw shape:   {raw.shape}")
    print(f"Clean shape: {clean.shape}")
    print(f"Class counts: {clean[config.TARGET_COLUMN].value_counts().sort_index().to_dict()}")
    print(f"Header names with leading/trailing whitespace (stripped): {len(whitespace_columns)}")
    print(f"Constant columns: {cleaning_report['constant_columns']}")
    print(f"Identical column pairs: {cleaning_report['duplicate_column_pairs']}")
    for drop in cleaning_report["proposed_drops"]:
        print(f"Dropped {drop['column']!r}: {drop['reason']}")
    print("Fold class counts:")
    print(fold_class_counts(assignments).to_string())
    for path in (
        config.CLEAN_CSV_PATH,
        config.DATA_QUALITY_PATH,
        config.DATA_PROFILE_PATH,
        config.FOLD_ASSIGNMENTS_PATH,
    ):
        print(f"Wrote {path.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
