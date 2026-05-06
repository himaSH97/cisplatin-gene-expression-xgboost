"""Print a concise profile of a Parquet dataset: shape, dtypes, columns, missing values.

Run from project root (optional name overrides the default)::

    python -m src.data.dataset_profile
    python -m src.data.dataset_profile cisplatin_final
    python -m src.data.dataset_profile data/processed/cisplatin.parquet

Edit ``DATASET_NAME`` below to change the default stem (loads
``data/processed/{DATASET_NAME}.parquet`` unless you pass a path or name on the CLI).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = _ROOT / "data" / "processed"

# --- default: Parquet stem under ``data/processed/`` (``.parquet`` added automatically) ---
DATASET_NAME = "cisplatin_final"

# How many column names to show from the start / end when the table is wide
_COLUMN_NAME_PREVIEW = 20
# Max columns to list that have missing values (sorted by missing count, descending)
_MISSING_COLUMNS_TO_SHOW = 50

META_COLUMNS = ("CELL_LINE_NAME", "DRUG_NAME")


def resolve_dataset_path(name_or_path: str) -> Path:
    """Resolve a dataset to a Parquet path.

    Tries, in order:

    1. ``name_or_path`` as a filesystem path (absolute or relative to CWD).
    2. ``<project_root> / name_or_path`` if that file exists.
    3. ``data/processed/{name}.parquet`` under the project root (``name`` may already
       include a ``.parquet`` suffix).
    """

    s = name_or_path.strip()
    if not s:
        raise ValueError("Dataset name or path is empty.")

    direct = Path(s).expanduser()
    if direct.is_file():
        return direct.resolve()

    under_root = (_ROOT / s).resolve()
    if under_root.is_file():
        return under_root

    stem = s if s.lower().endswith(".parquet") else f"{s}.parquet"
    processed = (PROCESSED_DIR / Path(stem).name).resolve()
    if processed.is_file():
        return processed

    raise FileNotFoundError(
        "Could not find a Parquet file for "
        f"{name_or_path!r}. Tried:\n"
        f"  - {direct}\n"
        f"  - {under_root}\n"
        f"  - {processed}"
    )


def profile_dataset(name_or_path: str | None = None) -> None:
    """Load a dataset by name (or path) and print ``profile_parquet`` output."""

    label = (name_or_path or DATASET_NAME).strip()
    path = resolve_dataset_path(label)
    print(f"Dataset: {label!r} → {path}\n")
    profile_parquet(path)


def _print_feature_float_audit(X: pd.DataFrame) -> None:
    """After dropping meta columns, count numeric vs. coercible-to-float feature columns."""

    n_feat = X.shape[1]
    print("=== Feature matrix (meta columns dropped) ===")
    print(f"  Shape: {X.shape[0]:,} rows × {n_feat:,} columns")
    if n_feat == 0:
        print("  (no feature columns left)")
        print()
        return

    already: list[str] = []
    convertible: list[str] = []
    not_convertible: list[str] = []

    non_numeric = [
        c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
    ]
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            already.append(c)

    for c in tqdm(non_numeric, desc="Check to_numeric", unit="col"):
        s = X[c]
        coerced = pd.to_numeric(s, errors="coerce")
        introduced = (s.notna() & coerced.isna()).sum()
        if introduced == 0:
            convertible.append(c)
        else:
            not_convertible.append(c)

    print()
    print("=== Float readiness (features only) ===")
    print(
        f"  Already numeric (int/float/etc.): {len(already):,} "
        f"({100 * len(already) / n_feat:.2f}% of features)"
    )
    print(
        f"  Non-numeric but fully parseable as float: {len(convertible):,} "
        f"({100 * len(convertible) / n_feat:.2f}%)"
    )
    print(
        f"  Would lose non-null values if coerced: {len(not_convertible):,} "
        f"({100 * len(not_convertible) / n_feat:.2f}%)"
    )
    print(
        f"  → Usable as float without data loss: "
        f"{len(already) + len(convertible):,} / {n_feat:,}"
    )
    if not_convertible:
        preview = not_convertible[:15]
        print(f"  Example non-convertible columns ({len(preview)} of {len(not_convertible)}):")
        for name in preview:
            print(f"    {name!r}")
        if len(not_convertible) > len(preview):
            print(f"    … +{len(not_convertible) - len(preview)} more …")
    print()


def _preview_column_names(cols: pd.Index) -> None:
    n = len(cols)
    print(f"  Total columns: {n}")
    if n == 0:
        return
    head_n = min(_COLUMN_NAME_PREVIEW, n)
    print(f"  First {head_n} names:")
    for c in cols[:head_n]:
        print(f"    {c!r}")
    if n > head_n:
        tail_n = min(_COLUMN_NAME_PREVIEW, n - head_n)
        if tail_n > 0:
            print(f"  Last {tail_n} names:")
            for c in cols[-tail_n:]:
                print(f"    {c!r}")
        if n > head_n + tail_n:
            print(f"  … omitted {n - head_n - tail_n} names in the middle …")


def profile_parquet(path: Path) -> None:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")

    print(f"File: {path}")
    print()

    df = pd.read_parquet(path, engine="pyarrow")
    n_rows, n_cols = df.shape
    print("=== Shape ===")
    print(f"  Rows:    {n_rows:,}")
    print(f"  Columns: {n_cols:,}")
    print()

    print("=== Dtypes ===")
    vc = df.dtypes.astype(str).value_counts()
    for dtype, count in vc.items():
        print(f"  {dtype}: {count:,}")
    print()

    print("=== Column names ===")
    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated()].unique().tolist()
        print(f"  WARNING: duplicate column labels ({len(dup)} examples): {dup[:10]}")
    _preview_column_names(df.columns)
    print()

    print("=== Index ===")
    print(f"  {df.index!r} (len={len(df.index):,})")
    print()

    print("=== Missing values ===")
    na_per_col = df.isna().sum()
    total_na = int(na_per_col.sum())
    n_cells = n_rows * n_cols if n_cols else 0
    pct = 100.0 * total_na / n_cells if n_cells else 0.0
    n_cols_with_na = int((na_per_col > 0).sum())
    print(f"  Total NaN cells:     {total_na:,}  ({pct:.4f}% of all cells)")
    print(f"  Columns with ≥1 NaN: {n_cols_with_na:,}")
    bad = na_per_col[na_per_col > 0].sort_values(ascending=False)
    if bad.empty:
        print("  No missing values in any column.")
    else:
        show = bad.head(_MISSING_COLUMNS_TO_SHOW)
        print(
            f"  Top {len(show)} columns by NaN count "
            f"(of {len(bad)} columns that have gaps):"
        )
        for col, k in show.items():
            pct_col = 100.0 * float(k) / n_rows if n_rows else 0.0
            print(f"    {k:>10,}  ({pct_col:5.1f}% of rows)  {col!r}")
        if len(bad) > len(show):
            print(f"  … {len(bad) - len(show)} more columns with NaN (not listed) …")
    print()

    print("=== Duplicate rows (full row) ===")
    ndup = int(df.duplicated().sum())
    print(f"  Count: {ndup:,}")
    print()

    meta = {"CELL_LINE_NAME", "DRUG_NAME", "LN_IC50"}
    present_meta = [c for c in meta if c in df.columns]
    if present_meta:
        print("=== Quick stats (key columns) ===")
        for c in present_meta:
            s = df[c]
            print(f"  {c}: dtype={s.dtype}")
            if c == "LN_IC50" and np.issubdtype(s.dtype, np.number):
                print(s.describe().to_string(header=False))
            elif c in ("CELL_LINE_NAME", "DRUG_NAME"):
                print(f"    nunique: {s.nunique():,}")
                print(f"    sample: {s.dropna().head(3).tolist()}")
        print()

    X = df.drop(columns=list(META_COLUMNS), errors="ignore")
    _print_feature_float_audit(X)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a Parquet dataset (default: DATASET_NAME under data/processed/).",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help=(
            "Dataset stem (e.g. cisplatin_final), filename (e.g. cisplatin.parquet), "
            "or path to a .parquet file. Default: DATASET_NAME in this module."
        ),
    )
    args = parser.parse_args()
    profile_dataset(args.dataset)


if __name__ == "__main__":
    main()
