"""Load numeric cisplatin Parquet and run validation."""

from pathlib import Path

import pandas as pd

from src.data.validate import validate_cisplatin_parquet

root = Path(__file__).resolve().parent
path = root / "data/processed/cisplatin.parquet"

df = pd.read_parquet(path, engine="pyarrow")
print("Loaded:", path)
print(df.head())
validate_cisplatin_parquet(df)
