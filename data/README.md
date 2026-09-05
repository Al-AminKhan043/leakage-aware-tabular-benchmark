# Data availability

The raw and cleaned dataset files (`thesis_dataset.csv`, `thesis_clean.csv`) are **not included** in this repository. This is an independently compiled industrial procurement dataset, and it is being held back pending publication so that authorship and provenance are properly established before wider release.

**Available on reasonable request** — contact the author (see main [README](../README.md)).

To run the notebooks yourself, place your own copies of the two files at:
```
data/thesis_dataset.csv
data/thesis_clean.csv
```
`01_leakage_diagnosis.ipynb` generates `thesis_clean.csv` from `thesis_dataset.csv`, so only the raw file is strictly required to start.

## Schema (for methodological transparency)

`thesis_dataset.csv` — 9,310 rows × 14 columns:

| Column | Type | Notes |
|---|---|---|
| `Date` | string (date) | Procurement date |
| `Chemical Name` | string (categorical) | |
| `Price/Kg` | integer | |
| `Price/MT` | integer | |
| `Packing` | integer (categorical code) | |
| `Purity` | integer | |
| `Chemical Nature` | string (categorical) | |
| `Supplier Country` | string (categorical) | |
| `Supplier Company` | string (categorical) | |
| `Storing Container` | string (categorical) | |
| `Boiling Point` | float | Missing for chemicals that decompose before boiling (see notebook 1 for imputation/sentinel handling) |
| `Composition` | string (categorical) | |
| `Usage` | string (categorical) | |
| `Quality` | string (binary target) | Good / Bad |

`thesis_clean.csv` adds one column, `signature` — a hash of all feature columns (excluding `Date` and `Quality`), used to detect and group duplicate feature vectors. See `01_leakage_diagnosis.ipynb` for the exact construction.
