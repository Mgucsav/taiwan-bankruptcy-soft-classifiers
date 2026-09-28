# Data directory

Nothing in `raw/` or `processed/` is tracked by Git (only the `.gitkeep` placeholders).
Every file is re-created by the scripts from the official UCI source.

| Path | Created by | Content |
|---|---|---|
| `raw/taiwanese_bankruptcy_prediction.zip` | `scripts/download_data.py` | Official UCI ZIP, unmodified |
| `raw/data.csv` | `scripts/download_data.py` | The only member extracted from the ZIP |
| `raw/download_metadata.json` | `scripts/download_data.py` | Source URL, UTC download time, SHA-256 digests |
| `processed/taiwan_bankruptcy_clean.csv` | `scripts/prepare_data.py` | 6819 rows × 93 columns (92 features + `Bankrupt?`) |

- Source: <https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction>
- DOI: <https://doi.org/10.24432/C5004D>
- License: CC BY 4.0 — the processed CSV is an adaptation (one constant column and two
  duplicate columns removed; column-name whitespace stripped). Credit the UCI source when
  sharing it.

The processed CSV is **not** scaled, resampled or feature-selected. Those steps are fitted
exclusively on each training fold.
