"""Filter the merged GDSC + expression Parquet parts to one drug and save a single file.

Edit FILTER_DRUG to match ``DRUG_NAME`` in the source data exactly (GDSC2 spelling).
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --- set the drug to extract (must match ``DRUG_NAME`` in the merged table) ---
FILTER_DRUG = "Cisplatin"

# Paths relative to project root
ROOT = Path(__file__).resolve().parent
MERGED_PARTS_DIR = ROOT / "data" / "processed" / "final_dataset_parquet"


def _output_path(drug: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in drug.strip()).strip("_").lower()
    return ROOT / "data" / "processed" / f"{safe}.parquet"


def main() -> None:
    part_paths = sorted(MERGED_PARTS_DIR.glob("part_*.parquet"))
    if not part_paths:
        raise FileNotFoundError(
            f"No part_*.parquet under {MERGED_PARTS_DIR}. Run main.py merge first."
        )

    chunks: list[pd.DataFrame] = []
    for path in tqdm(part_paths, desc=f"Filter {FILTER_DRUG!r}", unit="part"):
        df = pd.read_parquet(path, engine="pyarrow")
        sub = df[df["DRUG_NAME"] == FILTER_DRUG]
        if not sub.empty:
            chunks.append(sub)

    if not chunks:
        sample = pd.read_parquet(part_paths[0], engine="pyarrow", columns=["DRUG_NAME"])
        examples = sample["DRUG_NAME"].drop_duplicates().head(25).tolist()
        raise ValueError(
            f"No rows with DRUG_NAME == {FILTER_DRUG!r}. "
            f"Examples from first part: {examples}"
        )

    tqdm.write("Concatenating filtered rows…")
    out = pd.concat(chunks, ignore_index=True)
    out_path = _output_path(FILTER_DRUG)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Writing Parquet → {out_path.name}…")
    out.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

    print(f"Drug: {FILTER_DRUG!r}")
    print(f"Rows: {len(out)}, columns: {out.shape[1]}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
