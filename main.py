from pathlib import Path

from src.data.load_data import load_expression, load_ic50, load_mapping
from src.data.preprocess import clean_expression, clean_ic50, clean_mapping
from src.data.merge import (
    map_cell_lines,
    filter_common_cell_lines,
    merge_datasets_stream_to_parquet_dir,
)
from src.data.validate import sanity_check_streaming


def main():
    root = Path(__file__).resolve().parent

    # Load data (paths relative to project root so runs work from any CWD)
    expression = load_expression(str(root / "data/raw/rnaseq_merged_rsem_tpm_20260323.csv"))
    ic50 = load_ic50(str(root / "data/raw/gdsc2_ic50.csv"))
    mapping = load_mapping(str(root / "data/raw/model_list.csv"))

    print("Data loaded")

    # Preprocess
    expression = clean_expression(expression)
    ic50 = clean_ic50(ic50)
    mapping = clean_mapping(mapping)

    print("Data cleaned")

    # Map IDs
    expression = map_cell_lines(expression, mapping)

    print("Cell lines mapped")

    # Filter common samples
    expression, ic50 = filter_common_cell_lines(expression, ic50)

    print("Common cell lines:", len(expression))

    # Merge + write Parquet parts (keeps full merged frame out of memory).
    # Read: pd.read_parquet(out_parquet_dir, engine="pyarrow")
    # Folder is created immediately; part_*.parquet files appear after each batch (first is slowest).
    out_parquet_dir = str(root / "data/processed/final_dataset_parquet")
    n_rows, n_cols = merge_datasets_stream_to_parquet_dir(
        expression, ic50, out_parquet_dir
    )

    print("Datasets merged and written")

    sanity_check_streaming(expression, ic50, out_parquet_dir, n_rows, n_cols)

    print(f"Final dataset on disk: {n_rows} rows × {n_cols} cols → {out_parquet_dir}")

    print("Pipeline complete ✅")


if __name__ == "__main__":
    main()