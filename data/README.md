# Data availability

The raw and cleaned dataset files (`thesis_dataset.csv`, `thesis_clean.csv`) are not included in this repository. The data is proprietary to the participating textile-chemical firm. An anonymized version supporting the reported results is available from the author on reasonable request, subject to the firm's approval.

To run the notebooks yourself, place your own copies of the two files at:
```
data/thesis_dataset.csv
data/thesis_clean.csv
```
`01_leakage_diagnosis.ipynb` generates `thesis_clean.csv` from `thesis_dataset.csv`, so only the raw file is strictly required to start.

## Schema (for methodological transparency)

`thesis_dataset.csv` has 9,310 rows × 14 columns:

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

`thesis_clean.csv` adds one column, `signature`: the exact values of all 12 feature columns (excluding `Date` and `Quality`) joined with `|`, used to detect and group duplicate feature vectors. See `01_leakage_diagnosis.ipynb` for the exact construction.
