"""Confusion-matrix counts for the bankrupt (positive) class."""

from __future__ import annotations

import numpy as np

CONFUSION_COLUMNS: tuple[str, ...] = (
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> dict[str, int]:
    """Return TP, TN, FP and FN with ``positive`` as the positive label."""
    truth = np.asarray(y_true) == positive
    predicted = np.asarray(y_pred) == positive
    return {
        "true_positive": int(np.sum(truth & predicted)),
        "true_negative": int(np.sum(~truth & ~predicted)),
        "false_positive": int(np.sum(~truth & predicted)),
        "false_negative": int(np.sum(truth & ~predicted)),
    }
