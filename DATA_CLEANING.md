# Data Cleaning & Preprocessing Pipeline

## Overview

This document describes the data cleaning and preprocessing pipeline used to prepare gene expression and drug response data for the cisplatin drug response prediction model.

---

## Data Sources

### Raw Data Files
| File | Description | Location |
|------|-------------|----------|
| `rnaseq_merged_rsem_tpm_20260323.csv` | RNA-seq gene expression data (TPM normalized) | `data/raw/` |
| `gdsc2_ic50.csv` | GDSC2 drug response IC50 values | `data/raw/` |
| `model_list.csv` | Cell line ID mapping (SIDM → cell line names) | `data/raw/` |

---

## Pipeline Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Expression    │     │     IC50        │     │    Mapping      │
│   (RNA-seq)     │     │  (Drug resp.)   │     │  (Cell line ID) │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ clean_expression│     │   clean_ic50    │     │  clean_mapping  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    map_cell_lines       │
                    │  (SIDM → cell names)    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ filter_common_cell_lines│
                    │   (intersection only)   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  merge_datasets_stream  │
                    │   (to Parquet parts)    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Validation & QC       │
                    │    sanity_check()       │
                    └─────────────────────────┘
```

---

## Data Loading

```python
from src.data.load_data import load_expression, load_ic50, load_mapping

# Load raw data
expression = load_expression("data/raw/rnaseq_merged_rsem_tpm_20260323.csv")
ic50 = load_ic50("data/raw/gdsc2_ic50.csv")
mapping = load_mapping("data/raw/model_list.csv")
```

All files loaded using `pd.read_csv()` with `low_memory=False` for expression data to handle mixed types.

---

## Cleaning Functions

### 1. Expression Data Cleaning (`clean_expression`)

```python
def clean_expression(expression: pd.DataFrame) -> pd.DataFrame:
```

**Steps performed:**

| Step | Description | Code |
|------|-------------|------|
| 1 | Remove unnamed columns | `expression.loc[:, ~expression.columns.str.contains('^Unnamed')]` |
| 2 | Remove metadata rows | Filter out rows where `model_id` is in `{'model_name', 'data_source', 'gene_symbol'}` |
| 3 | Set gene column as index | `expression.set_index('model_id')` |
| 4 | Transpose matrix | Genes → columns, Samples → rows |
| 5 | Handle name collisions | Rename if `CELL_LINE_NAME` exists as gene feature |
| 6 | Reset index | Add `CELL_LINE_NAME` as column |
| 7 | Deduplicate column names | Append `__dup{n}` suffix to duplicate columns |

**Deduplication logic:**
```python
def _dedupe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    counts: Counter = Counter()
    new_cols: list[str] = []
    for col in df.columns:
        label = str(col)
        if counts[label] == 0:
            new_cols.append(label)
        else:
            new_cols.append(f"{label}__dup{counts[label]}")
        counts[label] += 1
    out = df.copy()
    out.columns = new_cols
    return out
```

### 2. IC50 Data Cleaning (`clean_ic50`)

```python
def clean_ic50(ic50: pd.DataFrame) -> pd.DataFrame:
    return ic50[['CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50']].copy()
```

**Keeps only essential columns:**
- `CELL_LINE_NAME` - Cell line identifier
- `DRUG_NAME` - Drug name
- `LN_IC50` - Natural log of IC50 (drug sensitivity)

### 3. Mapping Data Cleaning (`clean_mapping`)

```python
def clean_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    mapping.columns = mapping.columns.str.strip()  # Remove whitespace
    return mapping[['model_id', 'model_name']].copy()
```

**Steps:**
- Strip whitespace from column names
- Keep only `model_id` (SIDM ID) and `model_name` (cell line name)

---

## Data Integration

### Cell Line ID Mapping

```python
def map_cell_lines(expression: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    merged = expression.merge(
        mapping,
        left_on='CELL_LINE_NAME',
        right_on='model_id',
        how='inner'
    )
    merged['CELL_LINE_NAME'] = merged['model_name']
    merged = merged.drop(columns=['model_id', 'model_name'])
    return merged
```

Maps SIDM IDs (e.g., `SIDM00001`) to human-readable cell line names (e.g., `A549`).

### Filtering Common Cell Lines

```python
def filter_common_cell_lines(expression: pd.DataFrame, ic50: pd.DataFrame):
    common = set(expression['CELL_LINE_NAME']).intersection(
        set(ic50['CELL_LINE_NAME'])
    )
    expression = expression[expression['CELL_LINE_NAME'].isin(common)]
    ic50 = ic50[ic50['CELL_LINE_NAME'].isin(common)]
    return expression, ic50
```

Keeps only cell lines present in **both** expression and IC50 datasets.

---

## Merging & Output

### Streaming Merge to Parquet

```python
def merge_datasets_stream_to_parquet_dir(
    expression: pd.DataFrame,
    ic50: pd.DataFrame,
    out_dir: str,
    ic50_chunk_size: int = 2000,
    compression: str = "snappy",
) -> tuple[int, int]:
```

**Features:**
- **Memory efficient**: Processes IC50 in chunks (default 2000 rows)
- **Progress tracking**: tqdm progress bar + heartbeat logging
- **Snappy compression**: Fast read/write with good compression
- **Fault tolerant**: Creates `_FAILED.txt` on error, `_COMPLETE.txt` on success

**Output structure:**
```
data/processed/final_dataset_parquet/
├── part_00000.parquet
├── part_00001.parquet
├── part_00002.parquet
├── ...
└── _COMPLETE.txt
```

**Read merged data:**
```python
import pandas as pd
df = pd.read_parquet("data/processed/final_dataset_parquet", engine="pyarrow")
```

---

## Data Validation

### Sanity Checks Performed

```python
def sanity_check_streaming(expression, ic50, output_path, written_rows, written_cols):
```

| Check | Description |
|-------|-------------|
| Shape validation | Verify dimensions match expectations |
| Duplicate detection | Count duplicate rows |
| Missing value count | Total NaN values in dataset |
| Cell line matching | Identify unmatched cell lines between datasets |
| Drug distribution | Verify drug frequency distribution |
| IC50 statistics | Check LN_IC50 value range and distribution |

### Cisplatin-Specific Validation

```python
def validate_cisplatin_parquet(df: pd.DataFrame) -> None:
```

| Validation | Requirement |
|------------|-------------|
| Required columns | `CELL_LINE_NAME`, `DRUG_NAME`, `LN_IC50` must exist |
| No duplicate columns | Parquet rejects duplicate column names |
| Numeric features | All gene expression columns must be numeric |
| Key uniqueness | No duplicate `(CELL_LINE_NAME, DRUG_NAME)` pairs |

---

## Running the Pipeline

### Full Pipeline Execution

```python
python main.py
```

**Output:**
```
Data loaded
Data cleaned
Cell lines mapped
Common cell lines: 742
Merge + Parquet: 15 batches (~2000 IC50 rows), compression='snappy'
Datasets merged and written
Final dataset on disk: 742 rows × 41148 cols → data/processed/final_dataset_parquet
Pipeline complete ✅
```

### Individual Steps

```python
from src.data.load_data import load_expression, load_ic50, load_mapping
from src.data.preprocess import clean_expression, clean_ic50, clean_mapping
from src.data.merge import map_cell_lines, filter_common_cell_lines

# Load
expression = load_expression("data/raw/rnaseq_merged_rsem_tpm_20260323.csv")
ic50 = load_ic50("data/raw/gdsc2_ic50.csv")
mapping = load_mapping("data/raw/model_list.csv")

# Clean
expression = clean_expression(expression)
ic50 = clean_ic50(ic50)
mapping = clean_mapping(mapping)

# Map & Filter
expression = map_cell_lines(expression, mapping)
expression, ic50 = filter_common_cell_lines(expression, ic50)
```

---

## Data Quality Summary

### Final Dataset Statistics (Cisplatin)

| Metric | Value |
|--------|-------|
| **Rows** | 742 |
| **Columns** | 41,148 |
| **Gene features** | 41,145 |
| **Unique cell lines** | 742 |
| **Target variable** | `LN_IC50` |

### LN_IC50 Distribution
```
count    742.000000
mean       3.192064
std        1.845289
min       -1.730244
25%        1.807910
50%        3.068794
75%        4.425258
max        9.229988
```

---

## File Structure

```
drg-res/
├── data/
│   ├── raw/
│   │   ├── rnaseq_merged_rsem_tpm_20260323.csv
│   │   ├── gdsc2_ic50.csv
│   │   └── model_list.csv
│   └── processed/
│       ├── final_dataset_parquet/
│       │   ├── part_*.parquet
│       │   └── _COMPLETE.txt
│       └── cisplatin_final.parquet
├── src/
│   └── data/
│       ├── load_data.py      # Data loading functions
│       ├── preprocess.py     # Cleaning functions
│       ├── merge.py          # Merging & output functions
│       └── validate.py       # Validation functions
└── main.py                   # Pipeline orchestration
```

---

## Dependencies

```
pandas
numpy
pyarrow
tqdm
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Duplicate column names | `_dedupe_column_names()` appends `__dup{n}` suffix |
| Memory overflow on merge | Use `merge_datasets_stream_to_parquet_dir()` for chunked processing |
| SIDM ID format mismatch | Mapping file converts SIDM IDs to cell line names |
| Non-numeric features | `validate_cisplatin_parquet()` catches non-numeric columns |
| Incomplete merge (crash) | Check for `_FAILED.txt` in output directory |
