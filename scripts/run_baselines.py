"""Run pre-specified classical baselines on the fixed five-fold split."""

from __future__ import annotations

import json

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.baselines import evaluate_baselines, summarize_fold_results
from taiwan_soft_classifiers.data import load_clean_data, write_csv


def main() -> None:
    clean = load_clean_data()
    assignments = __import__("pandas").read_csv(config.FOLD_ASSIGNMENTS_PATH)
    fold_results = evaluate_baselines(clean, assignments)
    summary = summarize_fold_results(fold_results)

    fold_path = config.ARTIFACTS_DIR / "baseline_fold_results.csv"
    summary_path = config.ARTIFACTS_DIR / "baseline_summary.csv"
    manifest_path = config.ARTIFACTS_DIR / "baseline_protocol.json"
    write_csv(fold_results, fold_path)
    write_csv(summary, summary_path)
    manifest_path.write_text(
        json.dumps(
            {
                "status": "exploratory_pre_tuning_baselines",
                "folds": config.N_SPLITS,
                "random_state": config.RANDOM_STATE,
                "selection_rule": "Models and default/fixed parameters specified before results",
                "leakage_control": "Scaling fitted inside each training fold via sklearn Pipeline",
                "primary_metrics": ["pr_auc", "mcc", "balanced_accuracy", "recall_positive"],
                "warning": "No hyperparameter tuning; do not call these optimized models",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Wrote {fold_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
