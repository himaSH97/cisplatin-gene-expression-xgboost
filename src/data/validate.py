import glob
import os

import numpy as np
import pandas as pd


def validate_cisplatin_parquet(df: pd.DataFrame) -> None:
    """Sanity checks for the numeric cisplatin feature table (single Parquet file)."""

    print("\n🔍 CISPLATIN PARQUET VALIDATION\n")

    required = {"CELL_LINE_NAME", "DRUG_NAME", "LN_IC50"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names: {dups[:20]}…")

    print("Shape (rows × cols):", df.shape)
    print("Unique cell lines:", df["CELL_LINE_NAME"].nunique())
    print("Drug names:", df["DRUG_NAME"].value_counts().head())

    X = df.drop(columns=list(required))
    non_num = X.select_dtypes(exclude=[np.number])
    n_bad = non_num.shape[1]
    print(f"\nFeature columns (non meta): {X.shape[1]}")
    print(f"Non-numeric feature columns: {n_bad}")
    if n_bad:
        ex = non_num.columns[:10].tolist()
        raise TypeError(
            f"{n_bad} feature columns are still non-numeric (examples: {ex})"
        )

    print("\nLN_IC50:")
    print(df["LN_IC50"].describe())

    key_dupes = df.duplicated(subset=["CELL_LINE_NAME", "DRUG_NAME"]).sum()
    print("\nDuplicate rows (same cell line + drug):", int(key_dupes))

    na_ln = df["LN_IC50"].isna().sum()
    print("Missing LN_IC50:", int(na_ln))

    na_X = X.isna().sum().sum()
    print("Total NaN in feature matrix:", int(na_X))

    print("\n🔍 CISPLATIN VALIDATION END\n")


def sanity_check_streaming(
    expression,
    ic50,
    output_path: str,
    written_rows: int,
    written_cols: int,
    sample_rows: int = 100_000,
):
    """Checks when the merged table was written incrementally (not fully in memory).

    ``output_path`` may be a CSV file or a directory of ``part_*.parquet`` files.
    """

    print("\n🔍 SANITY CHECK (streamed output)\n")

    print("Expression shape:", expression.shape)
    print("IC50 shape:", ic50.shape)
    print(f"Written merged rows: {written_rows}, columns: {written_cols}")
    print("Output path:", output_path)

    unmatched_expr = set(expression["CELL_LINE_NAME"]) - set(ic50["CELL_LINE_NAME"])
    unmatched_ic50 = set(ic50["CELL_LINE_NAME"]) - set(expression["CELL_LINE_NAME"])
    print("\nUnmatched cell lines (expression only):", len(unmatched_expr))
    print("Unmatched cell lines (IC50 only):", len(unmatched_ic50))

    print("\nUnique cell lines (expression):", expression["CELL_LINE_NAME"].nunique())
    print("Unique cell lines (IC50):", ic50["CELL_LINE_NAME"].nunique())

    try:
        key_cols = ["CELL_LINE_NAME", "DRUG_NAME", "LN_IC50"]
        if os.path.isdir(output_path):
            part_files = sorted(glob.glob(os.path.join(output_path, "part_*.parquet")))
            if not part_files:
                raise FileNotFoundError("no part_*.parquet in directory")
            chunks = []
            remaining = sample_rows
            for pf in part_files:
                if remaining <= 0:
                    break
                chunk = pd.read_parquet(
                    pf, engine="pyarrow", columns=key_cols
                )
                chunks.append(chunk)
                remaining -= len(chunk)
            sample = pd.concat(chunks, ignore_index=True)
            if len(sample) > sample_rows:
                sample = sample.iloc[:sample_rows]
        else:
            sample = pd.read_csv(
                output_path,
                nrows=sample_rows,
                usecols=key_cols,
            )
        print(f"\nSample stats (first {len(sample)} rows on key columns):")
        print("Duplicate rows in sample:", sample.duplicated().sum())
        print("Missing in sample:", sample.isnull().sum().sum())
        print("\nTop drugs in sample:")
        print(sample["DRUG_NAME"].value_counts().head())
        print("\nLN_IC50 in sample:")
        print(sample["LN_IC50"].describe())
    except Exception as e:
        print("\n(Sample read skipped:", e, ")")

    print("\n🔍 SANITY CHECK END\n")


def sanity_check(expression, ic50, final_data):
    print("\n🔍 SANITY CHECK START\n")

    # 1. Shape checks
    print("Expression shape:", expression.shape)
    print("IC50 shape:", ic50.shape)
    print("Final merged shape:", final_data.shape)

    # 2. Check duplicates
    dupes = final_data.duplicated().sum()
    print("\nDuplicate rows in final dataset:", dupes)

    # 3. Check missing values
    missing = final_data.isnull().sum().sum()
    print("Total missing values:", missing)

    # 4. Unique cell lines
    print("\nUnique cell lines (expression):", expression['CELL_LINE_NAME'].nunique())
    print("Unique cell lines (IC50):", ic50['CELL_LINE_NAME'].nunique())
    print("Unique cell lines (final):", final_data['CELL_LINE_NAME'].nunique())

    # 5. Drug distribution
    print("\nTop drugs in dataset:")
    print(final_data['DRUG_NAME'].value_counts().head())

    # 6. IC50 stats
    print("\nIC50 stats:")
    print(final_data['LN_IC50'].describe())

    # 7. Check alignment
    unmatched_expr = set(expression['CELL_LINE_NAME']) - set(ic50['CELL_LINE_NAME'])
    unmatched_ic50 = set(ic50['CELL_LINE_NAME']) - set(expression['CELL_LINE_NAME'])

    print("\nUnmatched cell lines (expression only):", len(unmatched_expr))
    print("Unmatched cell lines (IC50 only):", len(unmatched_ic50))

    print("\n🔍 SANITY CHECK END\n")
