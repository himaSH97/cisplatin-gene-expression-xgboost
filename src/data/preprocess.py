from collections import Counter

import pandas as pd


def _dedupe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column labels are unique (Parquet / merges require this)."""

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


def clean_expression(expression: pd.DataFrame) -> pd.DataFrame:
    """Clean and reshape expression data"""

    # Remove unwanted columns
    expression = expression.loc[:, ~expression.columns.str.contains('^Unnamed')]

    # Drop matrix metadata rows (same column as genes; would become columns named
    # model_name etc. after transpose and collide with mapping merges)
    _meta = {'model_name', 'data_source', 'gene_symbol'}
    expression = expression[~expression['model_id'].isin(_meta)]

    # Set gene column as index
    expression = expression.set_index('model_id')

    # Transpose (genes → columns, samples → rows)
    expression = expression.T

    # If a gene id equals the sample column name we use next, reset_index would duplicate it.
    sid = "CELL_LINE_NAME"
    expression.index.name = sid
    if sid in expression.columns:
        expression = expression.rename(columns={sid: f"{sid}__gene_feature"})

    expression = expression.reset_index()

    # Duplicate model_id rows in the source matrix → duplicate column labels; Parquet rejects those.
    expression = _dedupe_column_names(expression)

    return expression


def clean_ic50(ic50: pd.DataFrame) -> pd.DataFrame:
    """Keep only required IC50 columns"""
    return ic50[['CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50']].copy()


def clean_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    """Clean mapping file"""

    # Remove whitespace issues
    mapping.columns = mapping.columns.str.strip()

    # Select required columns
    return mapping[['model_id', 'model_name']].copy()
