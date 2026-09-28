"""Project-wide constants: paths, data source metadata and the raw/clean data contracts."""

from __future__ import annotations

from pathlib import Path

RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Paths (resolved relative to the repository root; no user-specific paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

RAW_ZIP_NAME: str = "taiwanese_bankruptcy_prediction.zip"
RAW_CSV_NAME: str = "data.csv"
DOWNLOAD_METADATA_NAME: str = "download_metadata.json"
CLEAN_CSV_NAME: str = "taiwan_bankruptcy_clean.csv"

RAW_ZIP_PATH: Path = RAW_DIR / RAW_ZIP_NAME
RAW_CSV_PATH: Path = RAW_DIR / RAW_CSV_NAME
DOWNLOAD_METADATA_PATH: Path = RAW_DIR / DOWNLOAD_METADATA_NAME
CLEAN_CSV_PATH: Path = PROCESSED_DIR / CLEAN_CSV_NAME

DATA_QUALITY_PATH: Path = ARTIFACTS_DIR / "data_quality.json"
DATA_PROFILE_PATH: Path = ARTIFACTS_DIR / "data_profile.csv"
FOLD_ASSIGNMENTS_PATH: Path = ARTIFACTS_DIR / "fold_assignments.csv"

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
DATASET_NAME: str = "Taiwanese Bankruptcy Prediction"
UCI_PAGE_URL: str = "https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction"
UCI_ZIP_URL: str = (
    "https://archive.ics.uci.edu/static/public/572/taiwanese%2Bbankruptcy%2Bprediction.zip"
)
DATASET_DOI: str = "https://doi.org/10.24432/C5004D"
DATASET_LICENSE: str = "CC BY 4.0"
DATASET_LICENSE_URL: str = "https://creativecommons.org/licenses/by/4.0/"

# Upper bound for the uncompressed size of the extracted CSV (defends against zip bombs).
MAX_CSV_BYTES: int = 64 * 1024 * 1024

# ---------------------------------------------------------------------------
# Raw data contract
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "Bankrupt?"
EXPECTED_RAW_ROWS: int = 6819
EXPECTED_RAW_COLUMNS: int = 96
EXPECTED_RAW_FEATURES: int = 95
EXPECTED_CLASSES: frozenset[int] = frozenset({0, 1})
EXPECTED_CLASS_COUNTS: dict[int, int] = {0: 6599, 1: 220}
EXPECTED_MISSING_CELLS: int = 0
EXPECTED_DUPLICATE_ROWS: int = 0

# ---------------------------------------------------------------------------
# Cleaning rules
# ---------------------------------------------------------------------------
# Columns to drop are detected from the data (single unique value; exact numerical equality
# with an earlier feature). The constants below only describe the structure the real data is
# expected to show; if the detected structure differs, cleaning stops instead of forcing it.
EXPECTED_CONSTANT_COLUMNS: tuple[str, ...] = ("Net Income Flag",)
EXPECTED_DUPLICATE_DROP_COUNT: int = 2
# Never removed automatically; cleaning stops if the data mark it as constant or duplicate.
PROTECTED_COLUMNS: tuple[str, ...] = ("Liability-Assets Flag",)

# ---------------------------------------------------------------------------
# Clean data contract
# ---------------------------------------------------------------------------
EXPECTED_CLEAN_ROWS: int = 6819
EXPECTED_CLEAN_FEATURES: int = 92
EXPECTED_CLEAN_COLUMNS: int = 93

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
N_SPLITS: int = 5
SHUFFLE: bool = True
# Maximum allowed absolute deviation between a fold's bankruptcy rate and the overall rate.
FOLD_RATE_TOLERANCE: float = 0.005
