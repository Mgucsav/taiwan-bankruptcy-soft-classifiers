"""Compare freshly produced experiment outputs with the frozen canonical pilot results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from taiwan_soft_classifiers import config

CANONICAL_DIR: Path = config.PROJECT_ROOT / "reports" / "pilot_results"
MANIFEST_NAME: str = "SHA256SUMS.txt"
CANONICAL_FILES: tuple[str, ...] = (
    "baseline_fold_results.csv",
    "baseline_summary.csv",
    "baseline_protocol.json",
    "soft_classifier_fold_results.csv",
    "soft_classifier_summary.csv",
    "soft_classifier_protocol.json",
)
KEY_COLUMNS: tuple[str, ...] = ("model", "fold")

# ---------------------------------------------------------------------------
# Tolerance policy (reporting-level reproducibility)
# ---------------------------------------------------------------------------
# Tier 1 - exact: text and integer columns (model, fold, n_test, positive_test,
#   predicted_positive, true/false positives/negatives).
# Tier 2 - rtol = atol = 1e-12: every floating-point metric, including all soft-classifier
#   metrics, all threshold-based baseline metrics and the ROC-AUC/PR-AUC of every model
#   except Logistic-balanced.
# Tier 3 - rtol = 0, atol = 1e-3: only the ROC-AUC and PR-AUC of Logistic-balanced and the
#   mean/std derived from them. Logistic regression probability-score rankings showed
#   cross-runner numerical variation below 1e-3 while predicted labels, confusion counts and
#   threshold-based metrics remained identical (GitHub Actions runs 36425263045 and
#   36431187367, identical package versions, different Azure regions). These two values are
#   therefore numerically stable within the declared tolerance and are reported to two
#   decimals; they are not claimed to be bit-for-bit deterministic across CPUs.
# Runtime columns (elapsed_seconds*) are never compared.
RTOL: float = 1e-12
ATOL: float = 1e-12
SCORE_TOLERANT_MODELS: frozenset[str] = frozenset({"Logistic-balanced"})
SCORE_METRICS: frozenset[str] = frozenset(
    {"roc_auc", "pr_auc", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std"}
)
SCORE_RTOL: float = 0.0
SCORE_ATOL: float = 1e-3


def tolerance_for(model: str | None, column: str) -> tuple[float, float]:
    """(rtol, atol) for one floating-point cell under the declared policy."""
    if model in SCORE_TOLERANT_MODELS and column in SCORE_METRICS:
        return SCORE_RTOL, SCORE_ATOL
    return RTOL, ATOL


@dataclass(frozen=True)
class Difference:
    file: str
    location: str
    column: str
    expected: Any
    produced: Any
    abs_diff: float | None = None
    tolerance: str | None = None

    def __str__(self) -> str:
        text = (
            f"{self.file} | {self.location} | column={self.column} | "
            f"expected={self.expected!r} | produced={self.produced!r}"
        )
        if self.abs_diff is not None:
            text += f" | abs_diff={self.abs_diff:.3e}"
        if self.tolerance is not None:
            text += f" | tolerance={self.tolerance}"
        return text


def is_runtime_column(column: str) -> bool:
    """Runtime measurements and statistics derived from them are never compared."""
    return column.startswith("elapsed_seconds")


def _row_label(row: pd.Series, keys: list[str], index: int) -> str:
    if not keys:
        return f"row={index}"
    return ", ".join(f"{k}={row[k]}" for k in keys)


def compare_tables(
    expected: pd.DataFrame,
    produced: pd.DataFrame,
    name: str,
) -> list[Difference]:
    """Exact comparison of text/integer columns; tolerance policy for floating-point columns."""
    if list(expected.columns) != list(produced.columns):
        missing = [c for c in expected.columns if c not in produced.columns]
        extra = [c for c in produced.columns if c not in expected.columns]
        return (
            [
                Difference(
                    name,
                    "columns",
                    "<structure>",
                    list(expected.columns),
                    list(produced.columns),
                )
            ]
            + [Difference(name, "columns", c, "present", "missing") for c in missing]
            + [Difference(name, "columns", c, "absent", "present") for c in extra]
        )

    keys = [k for k in KEY_COLUMNS if k in expected.columns]
    differences: list[Difference] = []
    if len(expected) != len(produced):
        differences.append(Difference(name, "rows", "<row count>", len(expected), len(produced)))
    if keys:
        exp_keys = list(expected[keys].itertuples(index=False, name=None))
        pro_keys = list(produced[keys].itertuples(index=False, name=None))
        for key in [k for k in exp_keys if k not in pro_keys]:
            differences.append(Difference(name, str(key), "<row>", "present", "missing"))
        for key in [k for k in pro_keys if k not in exp_keys]:
            differences.append(Difference(name, str(key), "<row>", "absent", "present"))
        if not differences and exp_keys != pro_keys:
            differences.append(Difference(name, "rows", "<row order>", exp_keys, pro_keys))
    if differences:
        return differences

    for column in expected.columns:
        if is_runtime_column(column):
            continue
        exp_col, pro_col = expected[column], produced[column]
        exp_float = pd.api.types.is_float_dtype(exp_col)
        pro_float = pd.api.types.is_float_dtype(pro_col)
        if exp_float != pro_float or (
            pd.api.types.is_numeric_dtype(exp_col) != pd.api.types.is_numeric_dtype(pro_col)
        ):
            differences.append(
                Difference(name, "dtype", column, str(exp_col.dtype), str(pro_col.dtype))
            )
            continue
        if exp_float:
            a = exp_col.to_numpy(dtype=float)
            b = pro_col.to_numpy(dtype=float)
            models = expected["model"].tolist() if "model" in expected.columns else [None] * len(a)
            for i in range(len(a)):
                rtol, atol = tolerance_for(models[i], column)
                both_nan = np.isnan(a[i]) and np.isnan(b[i])
                if both_nan or abs(b[i] - a[i]) <= atol + rtol * abs(a[i]):
                    continue
                differences.append(
                    Difference(
                        name,
                        _row_label(expected.iloc[i], keys, i),
                        column,
                        float(a[i]),
                        float(b[i]),
                        float(abs(a[i] - b[i])),
                        f"rtol={rtol:g}, atol={atol:g}",
                    )
                )
        else:
            for i in range(len(expected)):
                if exp_col.iloc[i] != pro_col.iloc[i]:
                    abs_diff = None
                    if pd.api.types.is_numeric_dtype(exp_col):
                        abs_diff = float(abs(exp_col.iloc[i] - pro_col.iloc[i]))
                    differences.append(
                        Difference(
                            name,
                            _row_label(expected.iloc[i], keys, i),
                            column,
                            exp_col.iloc[i].item()
                            if hasattr(exp_col.iloc[i], "item")
                            else exp_col.iloc[i],
                            pro_col.iloc[i].item()
                            if hasattr(pro_col.iloc[i], "item")
                            else pro_col.iloc[i],
                            abs_diff,
                        )
                    )
    return differences


def compare_json(expected: Any, produced: Any, name: str, path: str = "$") -> list[Difference]:
    """Compare parsed JSON values recursively; report every differing key path."""
    if isinstance(expected, dict) and isinstance(produced, dict):
        differences: list[Difference] = []
        for key in sorted(set(expected) | set(produced)):
            child = f"{path}.{key}"
            if key not in produced:
                differences.append(Difference(name, child, key, expected[key], "<missing>"))
            elif key not in expected:
                differences.append(Difference(name, child, key, "<absent>", produced[key]))
            else:
                differences.extend(compare_json(expected[key], produced[key], name, child))
        return differences
    if isinstance(expected, list) and isinstance(produced, list) and len(expected) == len(produced):
        differences = []
        for i, (a, b) in enumerate(zip(expected, produced, strict=True)):
            differences.extend(compare_json(a, b, name, f"{path}[{i}]"))
        return differences
    if expected != produced or type(expected) is not type(produced):
        return [Difference(name, path, path.rsplit(".", 1)[-1], expected, produced)]
    return []


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", float_precision="round_trip")


def compare_file(name: str, canonical_dir: Path, produced_dir: Path) -> list[Difference]:
    expected_path, produced_path = canonical_dir / name, produced_dir / name
    for label, path in (("canonical", expected_path), ("produced", produced_path)):
        if not path.is_file():
            return [Difference(name, "file", "<file>", f"{label} file exists", "missing")]
    if name.endswith(".json"):
        return compare_json(
            json.loads(expected_path.read_text(encoding="utf-8")),
            json.loads(produced_path.read_text(encoding="utf-8")),
            name,
        )
    return compare_tables(read_table(expected_path), read_table(produced_path), name)


def verify(
    canonical_dir: Path = CANONICAL_DIR,
    produced_dir: Path = config.ARTIFACTS_DIR,
    names: tuple[str, ...] = CANONICAL_FILES,
) -> list[Difference]:
    differences: list[Difference] = []
    for name in names:
        differences.extend(compare_file(name, canonical_dir, produced_dir))
    return differences


# ---------------------------------------------------------------------------
# SHA-256 manifest of the canonical files
# ---------------------------------------------------------------------------
def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(canonical_dir: Path = CANONICAL_DIR) -> str:
    """``sha256sum``-style lines (``<hash>  <name>``) in alphabetical order of file name."""
    lines = [f"{sha256_bytes(canonical_dir / name)}  {name}" for name in sorted(CANONICAL_FILES)]
    return "\n".join(lines) + "\n"


def parse_manifest(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        if name in entries:
            raise ValueError(f"Duplicate manifest entry for {name}")
        entries[name] = digest
    return entries
