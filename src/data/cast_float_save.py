"""Cast every column except id strings to float and write ``cisplatin_final.parquet``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]

STRING_COLUMNS = ("CELL_LINE_NAME", "DRUG_NAME")
_TOP_NA_COLUMNS_TO_LIST = 30


def _feature_columns(df: pd.DataFrame, string_columns: tuple[str, ...]) -> list[str]:
    return [c for c in df.columns if c not in string_columns]


def _print_missingness_details(
    label: str,
    df: pd.DataFrame,
    string_columns: tuple[str, ...],
) -> int:
    """Print NaN stats; returns total NaN count in the full frame."""

    n_rows, n_cols = df.shape
    na_per = df.isna().sum()
    total_na = int(na_per.sum())
    n_cells = n_rows * n_cols if n_cols else 0
    pct = 100.0 * total_na / n_cells if n_cells else 0.0

    print(f"--- {label} ---")
    print(f"  Shape: {n_rows:,} × {n_cols:,}")
    print(f"  Total NaN cells (whole table): {total_na:,} ({pct:.4f}% of cells)")
    print("  String columns (unchanged by float cast):")
    for c in string_columns:
        if c not in df.columns:
            print(f"    {c!r}: (absent)")
            continue
        n = int(df[c].isna().sum())
        print(f"    {c!r}: {n:,} NaN ({100 * n / n_rows:.2f}% of rows)" if n_rows else f"    {c!r}: {n:,} NaN")

    feats = _feature_columns(df, string_columns)
    if not feats:
        print("  Feature columns: (none)")
        print()
        return total_na

    sub = df[feats]
    na_feat = sub.isna().sum()
    total_feat_na = int(na_feat.sum())
    n_feat_cells = n_rows * len(feats)
    pct_f = 100.0 * total_feat_na / n_feat_cells if n_feat_cells else 0.0
    cols_with_any = int((na_feat > 0).sum())
    print(f"  Feature columns only ({len(feats):,} cols):")
    print(f"    Total NaN in features: {total_feat_na:,} ({pct_f:.4f}% of feature cells)")
    print(f"    Columns with ≥1 NaN: {cols_with_any:,}")

    bad = na_feat[na_feat > 0].sort_values(ascending=False)
    if not bad.empty:
        show = bad.head(_TOP_NA_COLUMNS_TO_LIST)
        print(f"    Top {len(show)} feature columns by NaN count:")
        for col, k in show.items():
            pr = 100.0 * float(k) / n_rows if n_rows else 0.0
            print(f"      {k:>10,}  ({pr:5.1f}% rows)  {col!r}")
        if len(bad) > len(show):
            print(f"      … +{len(bad) - len(show)} more columns with NaN …")
    print()
    return total_na


def _print_coercion_new_nans(introduced: dict[str, int]) -> None:
    """Cells that were non-null before cast but NaN after ``to_numeric``."""

    print("--- Coercion: new NaNs (was non-null, became NaN after to_numeric) ---")
    if not introduced:
        print("  None: no values were lost to coercion.")
        print()
        return

    total_new = sum(introduced.values())
    print(f"  Columns affected: {len(introduced):,}")
    print(f"  Total newly missing cells: {total_new:,}")
    ranked = sorted(introduced.items(), key=lambda x: -x[1])[:_TOP_NA_COLUMNS_TO_LIST]
    print(f"  Top {len(ranked)} columns by newly introduced NaN count:")
    for col, k in ranked:
        print(f"    {k:>10,}  {col!r}")
    if len(introduced) > len(ranked):
        print(f"    … +{len(introduced) - len(ranked)} more columns …")
    print()


def cast_to_float_except_strings(
    df: pd.DataFrame,
    string_columns: tuple[str, ...] = STRING_COLUMNS,
    coercion_nan_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Return a copy where all columns except ``string_columns`` are float (via ``to_numeric``).

    If ``coercion_nan_counts`` is a dict, it is filled with ``{column: n_new}`` for each column
    where ``to_numeric`` turned a previously non-null cell into NaN.
    """

    out = df.copy()
    to_convert = [c for c in out.columns if c not in string_columns]
    for c in tqdm(to_convert, desc="Cast to float", unit="col"):
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coercion_nan_counts is not None:
            n_new = int((df[c].notna() & coerced.isna()).sum())
            if n_new:
                coercion_nan_counts[c] = n_new
        out[c] = coerced
    return out


def write_cisplatin_final(
    input_path: Path | None = None,
    output_path: Path | None = None,
    *,
    string_columns: tuple[str, ...] = STRING_COLUMNS,
    report: bool = True,
) -> Path:
    """Load Parquet, cast non-string columns to float, save as ``cisplatin_final.parquet``.

    When ``report`` is True, prints missing-value summaries before/after cast and any
    NaNs introduced by numeric coercion.
    """

    in_path = (input_path or _ROOT / "data" / "processed" / "cisplatin.parquet").resolve()
    out_path = (output_path or _ROOT / "data" / "processed" / "cisplatin_final.parquet").resolve()

    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    df = pd.read_parquet(in_path, engine="pyarrow")
    missing = [c for c in string_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required string columns: {missing}")

    if report:
        print("Missing-value report\n")
        _print_missingness_details("BEFORE cast (input Parquet)", df, string_columns)

    coercion_nans: dict[str, int] = {}
    out = cast_to_float_except_strings(
        df,
        string_columns=string_columns,
        coercion_nan_counts=coercion_nans if report else None,
    )

    if report:
        _print_coercion_new_nans(coercion_nans)
        _print_missingness_details("AFTER cast (about to write)", out, string_columns)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

    print(f"Read:  {in_path}")
    print(f"Wrote: {out_path}  shape={out.shape}")
    return out_path


def main() -> None:
    write_cisplatin_final()


if __name__ == "__main__":
    main()
