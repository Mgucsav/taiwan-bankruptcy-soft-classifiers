"""Reproducible data pipeline for corporate bankruptcy prediction on Taiwan Stock Exchange data.

Classifiers (FPFS-kNN, FPFS-AC, IFPIFS-HC, IFPIFSC, PFS-kNN) are not implemented yet.
"""

from taiwan_soft_classifiers.config import RANDOM_STATE

__version__ = "0.1.0"

__all__ = ["RANDOM_STATE", "__version__"]
