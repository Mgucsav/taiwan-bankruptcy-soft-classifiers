"""Shared fixtures. Tests read the locally downloaded data and never access the network."""

from __future__ import annotations

import pandas as pd
import pytest

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.data import clean_data, load_raw_data


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    if not config.RAW_CSV_PATH.is_file():
        pytest.skip("Raw data missing; run `python scripts/download_data.py` first")
    return load_raw_data(config.RAW_CSV_PATH)


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    clean, _ = clean_data(raw_df)
    return clean
