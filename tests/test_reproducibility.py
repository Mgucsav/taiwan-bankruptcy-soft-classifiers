"""The canonical-results gate (scripts/verify_reproducibility.py) and the SHA-256 manifest."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.reproducibility import (
    CANONICAL_DIR,
    CANONICAL_FILES,
    MANIFEST_NAME,
    compare_json,
    compare_tables,
    parse_manifest,
    read_table,
    sha256_bytes,
    verify,
)

FOLD_FILE = "soft_classifier_fold_results.csv"


@pytest.fixture
def fold_table() -> pd.DataFrame:
    return read_table(CANONICAL_DIR / FOLD_FILE)


@pytest.fixture
def produced_dir(tmp_path: Path) -> Path:
    """A copy of the canonical files standing in for freshly produced artifacts."""
    for name in CANONICAL_FILES:
        shutil.copyfile(CANONICAL_DIR / name, tmp_path / name)
    return tmp_path


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def test_equal_tables_pass(fold_table: pd.DataFrame) -> None:
    assert compare_tables(fold_table, fold_table.copy(), FOLD_FILE) == []


def test_float_difference_within_tolerance_passes(fold_table: pd.DataFrame) -> None:
    produced = fold_table.copy()
    produced["mcc"] = produced["mcc"] * (1 + 1e-14)
    produced.loc[0, "balanced_accuracy"] += 5e-13
    assert compare_tables(fold_table, produced, FOLD_FILE) == []


def test_float_difference_outside_tolerance_fails(fold_table: pd.DataFrame) -> None:
    produced = fold_table.copy()
    produced.loc[3, "recall_positive"] += 1e-9
    differences = compare_tables(fold_table, produced, FOLD_FILE)
    assert len(differences) == 1
    diff = differences[0]
    assert diff.column == "recall_positive"
    assert diff.location == f"model={fold_table.loc[3, 'model']}, fold={fold_table.loc[3, 'fold']}"
    assert diff.expected == fold_table.loc[3, "recall_positive"]
    assert diff.produced == produced.loc[3, "recall_positive"]
    assert diff.abs_diff == pytest.approx(1e-9, rel=1e-6)
    text = str(diff)
    for part in (
        "model=",
        "fold=",
        "column=recall_positive",
        "expected=",
        "produced=",
        "abs_diff=",
    ):
        assert part in text


def test_integer_difference_fails(fold_table: pd.DataFrame) -> None:
    produced = fold_table.copy()
    produced.loc[0, "true_positive"] += 1
    differences = compare_tables(fold_table, produced, FOLD_FILE)
    assert [d.column for d in differences] == ["true_positive"]
    assert differences[0].abs_diff == 1.0


def test_text_difference_fails(fold_table: pd.DataFrame) -> None:
    table = read_table(CANONICAL_DIR / "baseline_summary.csv")
    produced = table.copy()
    produced.loc[1, "model"] = "Logistic-balanced-renamed"
    differences = compare_tables(table, produced, "baseline_summary.csv")
    assert differences
    assert {d.column for d in differences} == {"<row>"}


def test_missing_and_extra_rows_fail(fold_table: pd.DataFrame) -> None:
    missing = compare_tables(fold_table, fold_table.iloc[:-1].copy(), FOLD_FILE)
    assert any(d.column == "<row count>" for d in missing)
    assert any(d.produced == "missing" for d in missing)
    extra_row = fold_table.iloc[[0]].assign(fold=99)
    extra = compare_tables(fold_table, pd.concat([fold_table, extra_row]), FOLD_FILE)
    assert any(d.produced == "present" and "99" in d.location for d in extra)


def test_row_order_change_fails(fold_table: pd.DataFrame) -> None:
    shuffled = fold_table.iloc[::-1].reset_index(drop=True)
    differences = compare_tables(fold_table, shuffled, FOLD_FILE)
    assert [d.column for d in differences] == ["<row order>"]


def test_missing_and_extra_columns_fail(fold_table: pd.DataFrame) -> None:
    missing = compare_tables(fold_table, fold_table.drop(columns=["mcc"]), FOLD_FILE)
    assert any(d.column == "mcc" and d.produced == "missing" for d in missing)
    extra = compare_tables(fold_table, fold_table.assign(roc_auc=0.5), FOLD_FILE)
    assert any(d.column == "roc_auc" and d.produced == "present" for d in extra)


def test_dtype_change_fails(fold_table: pd.DataFrame) -> None:
    produced = fold_table.copy()
    produced["true_positive"] = produced["true_positive"].astype(float)
    differences = compare_tables(fold_table, produced, FOLD_FILE)
    assert [(d.location, d.column) for d in differences] == [("dtype", "true_positive")]


def test_elapsed_seconds_differences_are_ignored(fold_table: pd.DataFrame) -> None:
    produced = fold_table.copy()
    produced["elapsed_seconds"] = produced["elapsed_seconds"] * 7 + 100
    assert compare_tables(fold_table, produced, FOLD_FILE) == []
    summary = read_table(CANONICAL_DIR / "soft_classifier_summary.csv")
    changed = summary.assign(
        elapsed_seconds_mean=summary["elapsed_seconds_mean"] + 1,
        elapsed_seconds_std=summary["elapsed_seconds_std"] * 3,
    )
    assert compare_tables(summary, changed, "soft_classifier_summary.csv") == []


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def test_json_equal_dicts_pass_regardless_of_key_order() -> None:
    a = {"x": 1, "nested": {"k": 3, "lambda": 0.5}, "list": [1, 2]}
    b = {"list": [1, 2], "nested": {"lambda": 0.5, "k": 3}, "x": 1}
    assert compare_json(a, b, "p.json") == []


def test_json_value_and_key_differences_fail() -> None:
    expected = json.loads((CANONICAL_DIR / "soft_classifier_protocol.json").read_text("utf-8"))
    changed = json.loads(json.dumps(expected))
    changed["hyperparameters"]["PFS-kNN"]["k"] = 5
    del changed["tuning"]
    changed["extra"] = True
    differences = compare_json(expected, changed, "soft_classifier_protocol.json")
    locations = {d.location for d in differences}
    assert locations == {"$.hyperparameters.PFS-kNN.k", "$.tuning", "$.extra"}
    k_diff = next(d for d in differences if d.location.endswith(".k"))
    assert (k_diff.expected, k_diff.produced) == (3, 5)


def test_json_type_difference_fails() -> None:
    assert compare_json({"k": 3}, {"k": 3.0}, "p.json")
    assert compare_json({"flag": 1}, {"flag": True}, "p.json")


# ---------------------------------------------------------------------------
# Whole-directory gate and script
# ---------------------------------------------------------------------------
def test_canonical_files_reproduce_themselves(produced_dir: Path) -> None:
    assert verify(CANONICAL_DIR, produced_dir) == []


def test_missing_produced_file_fails(produced_dir: Path) -> None:
    (produced_dir / "baseline_summary.csv").unlink()
    differences = verify(CANONICAL_DIR, produced_dir)
    assert [(d.file, d.produced) for d in differences] == [("baseline_summary.csv", "missing")]


def _load_script():
    path = config.PROJECT_ROOT / "scripts" / "verify_reproducibility.py"
    spec = importlib.util.spec_from_file_location("verify_reproducibility", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_success_and_failure(produced_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script()
    assert script.main(produced_dir=produced_dir) == 0
    assert "canonical results reproduced" in capsys.readouterr().out

    table = read_table(produced_dir / FOLD_FILE)
    table.loc[2, "mcc"] += 0.01
    table.to_csv(produced_dir / FOLD_FILE, index=False)
    assert script.main(produced_dir=produced_dir) == 1
    captured = capsys.readouterr()
    assert "canonical results reproduced" not in captured.out
    assert "column=mcc" in captured.err
    assert "abs_diff=" in captured.err


# ---------------------------------------------------------------------------
# SHA-256 manifest
# ---------------------------------------------------------------------------
def manifest_entries() -> dict[str, str]:
    return parse_manifest((CANONICAL_DIR / MANIFEST_NAME).read_text(encoding="ascii"))


def test_manifest_lists_exactly_the_canonical_files() -> None:
    entries = manifest_entries()
    assert set(entries) == set(CANONICAL_FILES)
    assert MANIFEST_NAME not in entries
    names = [
        line.split("  ", 1)[1]
        for line in (CANONICAL_DIR / MANIFEST_NAME).read_text("ascii").splitlines()
    ]
    assert names == sorted(names)


@pytest.mark.parametrize("name", sorted(CANONICAL_FILES))
def test_manifest_hash_matches_file(name: str) -> None:
    path = CANONICAL_DIR / name
    assert path.is_file()
    assert sha256_bytes(path) == manifest_entries()[name]


def test_manifest_rejects_duplicate_entries() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        parse_manifest("aa  x.csv\nbb  x.csv\n")
