"""Check that the experiment outputs in artifacts/ reproduce the canonical pilot results.

Usage:
    python scripts/verify_reproducibility.py

Compares every file in reports/pilot_results/ listed in ``CANONICAL_FILES`` with the file of the
same name in artifacts/. Text and integer columns must match exactly, floating-point columns
within rtol = atol = 1e-12; runtime columns (``elapsed_seconds*``) are ignored. JSON files are
compared as parsed objects. Exits with code 1 and lists every difference otherwise.
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
    print(f"Tolerance: rtol={RTOL:g}, atol={ATOL:g}; elapsed_seconds* ignored")
    if differences:
        print(f"FAILED: {len(differences)} difference(s)", file=sys.stderr)
        for difference in differences:
            print(f"  {difference}", file=sys.stderr)
        return 1
    for name in CANONICAL_FILES:
        print(f"  OK {name}")
    print("canonical results reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
