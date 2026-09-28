"""Check that the experiment outputs in artifacts/ reproduce the canonical pilot results.

Usage:
    python scripts/verify_reproducibility.py

Compares every file in reports/pilot_results/ listed in ``CANONICAL_FILES`` with the file of the
same name in artifacts/ under the declared tolerance policy (see
``taiwan_soft_classifiers.reproducibility``):

- text and integer columns (model, fold, counts, TP/TN/FP/FN): exact;
- floating-point metrics, including all soft-classifier metrics: rtol = atol = 1e-12;
- ROC-AUC/PR-AUC of Logistic-balanced and their mean/std only: rtol = 0, atol = 1e-3;
- runtime columns (``elapsed_seconds*``): ignored.

JSON files are compared as parsed objects. Exits with code 1 and lists every difference
otherwise. This is reporting-level reproducibility, not a bit-for-bit determinism claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.reproducibility import (
    ATOL,
    CANONICAL_DIR,
    CANONICAL_FILES,
    RTOL,
    SCORE_ATOL,
    SCORE_RTOL,
    SCORE_TOLERANT_MODELS,
    verify,
)


def _display(path: Path) -> Path:
    try:
        return path.relative_to(config.PROJECT_ROOT)
    except ValueError:
        return path


def main(produced_dir: Path = config.ARTIFACTS_DIR, canonical_dir: Path = CANONICAL_DIR) -> int:
    differences = verify(canonical_dir, produced_dir, CANONICAL_FILES)
    print(f"Canonical: {_display(canonical_dir)}")
    print(f"Produced:  {_display(produced_dir)}")
    print("Tolerance policy:")
    print("  text/integer columns (incl. TP/TN/FP/FN): exact")
    print(f"  floating-point metrics: rtol={RTOL:g}, atol={ATOL:g}")
    print(
        f"  {', '.join(sorted(SCORE_TOLERANT_MODELS))} ROC-AUC/PR-AUC (+ mean/std): "
        f"rtol={SCORE_RTOL:g}, atol={SCORE_ATOL:g}"
    )
    print("  elapsed_seconds*: ignored")
    if differences:
        print(f"FAILED: {len(differences)} difference(s)", file=sys.stderr)
        for difference in differences:
            print(f"  {difference}", file=sys.stderr)
        return 1
    for name in CANONICAL_FILES:
        print(f"  OK {name}")
    print("canonical results reproduced (within the declared tolerance policy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
