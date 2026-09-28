"""Run the baseline experiment three times on the same runner and compare the results.

Usage:
    python scripts/diagnose_baseline_determinism.py [--repeats 3]

Each repeat runs ``scripts/run_baselines.py`` in a fresh process with the same data, folds and
model configuration, writing to its own temporary directory outside the repository. The
existing files in artifacts/ are not touched; only artifacts/baseline_determinism.json is
written. The report states whether the Logistic-balanced ROC-AUC and PR-AUC values are
bit-for-bit identical across the repeats.

Exit code 1 if the repeats differ beyond the declared tolerance policy of
``taiwan_soft_classifiers.reproducibility`` (exact bit-level differences are only reported).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.data import write_json
from taiwan_soft_classifiers.reproducibility import CANONICAL_DIR, compare_tables, read_table

FOLD_FILE = "baseline_fold_results.csv"
SUMMARY_FILE = "baseline_summary.csv"
LOGISTIC = "Logistic-balanced"
REPORT_PATH = config.ARTIFACTS_DIR / "baseline_determinism.json"


def run_once(output_dir: Path) -> None:
    command = [
        sys.executable,
        str(config.PROJECT_ROOT / "scripts" / "run_baselines.py"),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True, cwd=config.PROJECT_ROOT, stdout=subprocess.DEVNULL)


def logistic_scores(folds: pd.DataFrame, summary: pd.DataFrame) -> dict[str, object]:
    rows = folds[folds["model"] == LOGISTIC].sort_values("fold")
    mean = summary.set_index("model").loc[LOGISTIC]
    return {
        "roc_auc_by_fold": [float(v) for v in rows["roc_auc"]],
        "pr_auc_by_fold": [float(v) for v in rows["pr_auc"]],
        "roc_auc_mean": float(mean["roc_auc_mean"]),
        "pr_auc_mean": float(mean["pr_auc_mean"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repeat the baselines and compare the runs.")
    parser.add_argument("--repeats", type=int, default=3)
    repeats = parser.parse_args(argv).repeats

    folds: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="baseline-determinism-") as tmp:
        for i in range(1, repeats + 1):
            out = Path(tmp) / f"run_{i}"
            run_once(out)
            folds.append(read_table(out / FOLD_FILE))
            summaries.append(read_table(out / SUMMARY_FILE))
            print(f"repeat {i}: done")

    exact = all(f.equals(folds[0]) for f in folds[1:]) and all(
        s.equals(summaries[0]) for s in summaries[1:]
    )
    scores = [logistic_scores(f, s) for f, s in zip(folds, summaries, strict=True)]
    logistic_identical = all(s == scores[0] for s in scores[1:])

    within_runner = []
    for i in range(1, repeats):
        within_runner += [str(d) for d in compare_tables(folds[0], folds[i], FOLD_FILE)]
        within_runner += [str(d) for d in compare_tables(summaries[0], summaries[i], SUMMARY_FILE)]
    versus_canonical = [
        str(d) for d in compare_tables(read_table(CANONICAL_DIR / FOLD_FILE), folds[0], FOLD_FILE)
    ] + [
        str(d)
        for d in compare_tables(
            read_table(CANONICAL_DIR / SUMMARY_FILE), summaries[0], SUMMARY_FILE
        )
    ]

    report = {
        "repeats": repeats,
        "all_outputs_bitwise_identical": exact,
        "logistic_auc_bitwise_identical": logistic_identical,
        "logistic_scores_by_repeat": scores,
        "within_runner_policy_differences": within_runner,
        "repeat_1_vs_canonical_policy_differences": versus_canonical,
    }
    write_json(report, REPORT_PATH)

    print(f"All baseline outputs bit-for-bit identical across {repeats} repeats: {exact}")
    print(f"Logistic-balanced ROC-AUC/PR-AUC bit-for-bit identical: {logistic_identical}")
    for i, s in enumerate(scores, start=1):
        print(
            f"repeat {i}: roc_auc_mean={s['roc_auc_mean']!r} pr_auc_mean={s['pr_auc_mean']!r} "
            f"roc_auc_by_fold={s['roc_auc_by_fold']} pr_auc_by_fold={s['pr_auc_by_fold']}"
        )
    print(f"Repeat 1 vs canonical (policy): {len(versus_canonical)} difference(s)")
    for line in versus_canonical:
        print(f"  {line}")
    print(f"Wrote {REPORT_PATH.relative_to(config.PROJECT_ROOT)}")
    if within_runner:
        print("FAILED: repeats differ beyond the declared tolerance policy:", file=sys.stderr)
        for line in within_runner:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
