import os
import threading
import time
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm


def _heartbeat_worker(stop: threading.Event, label: str, interval_sec: float) -> None:
    """Print elapsed time while stop is not set (merge/write has no native progress)."""
    t0 = time.time()
    while not stop.wait(interval_sec):
        elapsed = int(time.time() - t0)
        tqdm.write(f"  … {elapsed}s in current step — {label}")


def map_cell_lines(expression: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Map SIDM IDs to cell line names"""

    merged = expression.merge(
        mapping,
        left_on='CELL_LINE_NAME',
        right_on='model_id',
        how='inner'
    )

    # Replace SIDM IDs with actual names
    merged['CELL_LINE_NAME'] = merged['model_name']

    # Drop helper columns
    merged = merged.drop(columns=['model_id', 'model_name'])

    return merged


def filter_common_cell_lines(expression: pd.DataFrame, ic50: pd.DataFrame):
    """Keep only common cell lines"""

    common = set(expression['CELL_LINE_NAME']).intersection(
        set(ic50['CELL_LINE_NAME'])
    )

    expression = expression[expression['CELL_LINE_NAME'].isin(common)]
    ic50 = ic50[ic50['CELL_LINE_NAME'].isin(common)]

    return expression, ic50


def merge_datasets(
    expression: pd.DataFrame,
    ic50: pd.DataFrame,
    ic50_chunk_size: int = 5000,
) -> pd.DataFrame:
    """Merge expression and IC50 data (chunked IC50 so tqdm can show progress)."""

    n = len(ic50)
    if n == 0:
        return pd.merge(expression, ic50, on='CELL_LINE_NAME')

    n_chunks = max(1, (n + ic50_chunk_size - 1) // ic50_chunk_size)
    row_batches = np.array_split(np.arange(n), n_chunks)
    parts: list[pd.DataFrame] = []
    for batch_idx in tqdm(row_batches, desc="Merge expression + IC50", unit="batch"):
        part = pd.merge(expression, ic50.iloc[batch_idx], on='CELL_LINE_NAME')
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def merge_datasets_stream_to_csv(
    expression: pd.DataFrame,
    ic50: pd.DataFrame,
    path: str,
    ic50_chunk_size: int = 2000,
    csv_write_rows: int = 250,
    heartbeat_sec: float = 15.0,
) -> tuple[int, int]:
    """Merge expression + IC50 in IC50 row batches and append each batch to CSV.

    Avoids holding the full merged DataFrame in memory. Returns (row_count, n_columns).

    The bar only advances after each batch finishes. While pandas merge/write runs,
    a background heartbeat prints every ``heartbeat_sec`` so long steps are not silent.

    * ``ic50_chunk_size`` — IC50 rows merged per batch. Larger → fewer merges, usually
      **faster end-to-end** (less Python/pandas overhead); smaller → more tqdm updates.
    * ``csv_write_rows`` — merged rows written per ``to_csv`` call. Smaller → lower
      **peak RAM** during text serialization on very wide tables; slightly more I/O overhead.
      Use ``0`` to write each merged batch in one shot (old behaviour).
    """

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n = len(ic50)
    if n == 0:
        empty = pd.merge(expression, ic50, on="CELL_LINE_NAME")
        empty.to_csv(path, index=False)
        return len(empty), empty.shape[1]

    n_chunks = max(1, (n + ic50_chunk_size - 1) // ic50_chunk_size)
    row_batches = np.array_split(np.arange(n), n_chunks)

    tqdm.write(
        f"Merge + write: {n_chunks} IC50 batches (~{ic50_chunk_size} rows/batch), "
        f"CSV slices of {csv_write_rows if csv_write_rows > 0 else 'all'} data rows. "
        f"Heartbeat every {heartbeat_sec:g}s; bar % after each IC50 batch."
    )

    total_rows = 0
    n_cols = 0
    first = True

    with open(path, "w", newline="", encoding="utf-8") as f:
        with tqdm(
            total=n_chunks,
            desc="Merge + write CSV",
            unit="batch",
            mininterval=0.2,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for i, batch_idx in enumerate(row_batches):
                r0, r1 = int(batch_idx[0]), int(batch_idx[-1]) + 1
                label = f"batch {i + 1}/{n_chunks} IC50[{r0}:{r1})"
                pbar.set_postfix_str(f"{label} merge…", refresh=False)
                pbar.refresh()

                stop_hb = threading.Event()
                hb = threading.Thread(
                    target=_heartbeat_worker,
                    args=(stop_hb, label, heartbeat_sec),
                    daemon=True,
                )
                hb.start()
                try:
                    part = pd.merge(
                        expression, ic50.iloc[batch_idx], on="CELL_LINE_NAME"
                    )

                    pbar.set_postfix_str(f"{label} write CSV…", refresh=False)
                    pbar.refresh()

                    if n_cols == 0:
                        n_cols = part.shape[1]

                    n_part = len(part)
                    if csv_write_rows <= 0:
                        part.to_csv(f, index=False, header=first)
                        first = False
                    else:
                        for wstart in range(0, n_part, csv_write_rows):
                            wend = min(wstart + csv_write_rows, n_part)
                            part.iloc[wstart:wend].to_csv(
                                f, index=False, header=first
                            )
                            first = False
                finally:
                    stop_hb.set()
                    hb.join(timeout=2.0)

                total_rows += len(part)
                del part

                pbar.update(1)

    return total_rows, n_cols


def merge_datasets_stream_to_parquet_dir(
    expression: pd.DataFrame,
    ic50: pd.DataFrame,
    out_dir: str,
    ic50_chunk_size: int = 2000,
    heartbeat_sec: float = 15.0,
    compression: str = "snappy",
) -> tuple[int, int]:
    """Merge + write one Parquet file per IC50 batch under ``out_dir``.

    Returns ``(total_row_count, n_columns)``. Load with e.g.
    ``pd.read_parquet(out_dir, engine="pyarrow")`` (same schema in all parts).

    The output directory is created immediately; ``part_*.parquet`` files only
    appear **after** each merge batch completes (the first batch is often very slow).
    """

    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Parquet output requires pyarrow. Install with: pip install pyarrow"
        ) from e

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    progress_marker = os.path.join(out_dir, "_MERGING_in_progress.txt")
    with open(progress_marker, "w", encoding="utf-8") as mf:
        mf.write(
            "Merge running. part_00000.parquet appears only after the FIRST batch "
            "finishes (wide matrix = can take many minutes).\n"
        )

    n = len(ic50)
    if n == 0:
        empty = pd.merge(expression, ic50, on="CELL_LINE_NAME")
        empty.to_parquet(
            os.path.join(out_dir, "part_00000.parquet"),
            index=False,
            engine="pyarrow",
            compression=compression,
        )
        try:
            os.remove(progress_marker)
        except OSError:
            pass
        with open(
            os.path.join(out_dir, "_COMPLETE.txt"), "w", encoding="utf-8"
        ) as cf:
            cf.write(f"rows={len(empty)} cols={empty.shape[1]}\n")
        return len(empty), empty.shape[1]

    n_chunks = max(1, (n + ic50_chunk_size - 1) // ic50_chunk_size)
    row_batches = np.array_split(np.arange(n), n_chunks)

    tqdm.write(
        f"Merge + Parquet: {n_chunks} batches (~{ic50_chunk_size} IC50 rows), "
        f"compression={compression!r} → {out_dir!r}. "
        f"Heartbeat every {heartbeat_sec:g}s."
    )

    total_rows = 0
    n_cols = 0
    failed_path = os.path.join(out_dir, "_FAILED.txt")
    complete_path = os.path.join(out_dir, "_COMPLETE.txt")

    try:
        if os.path.isfile(failed_path):
            os.remove(failed_path)
    except OSError:
        pass

    try:
        with tqdm(
            total=n_chunks,
            desc="Merge + write Parquet",
            unit="batch",
            mininterval=0.2,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            for i, batch_idx in enumerate(row_batches):
                r0, r1 = int(batch_idx[0]), int(batch_idx[-1]) + 1
                label = f"batch {i + 1}/{n_chunks} IC50[{r0}:{r1})"
                pbar.set_postfix_str(f"{label} merge…", refresh=False)
                pbar.refresh()

                stop_hb = threading.Event()
                hb = threading.Thread(
                    target=_heartbeat_worker,
                    args=(stop_hb, label, heartbeat_sec),
                    daemon=True,
                )
                hb.start()
                try:
                    part = pd.merge(
                        expression, ic50.iloc[batch_idx], on="CELL_LINE_NAME"
                    )

                    pbar.set_postfix_str(f"{label} write Parquet…", refresh=False)
                    pbar.refresh()

                    if n_cols == 0:
                        n_cols = part.shape[1]

                    part_path = os.path.join(out_dir, f"part_{i:05d}.parquet")
                    part.to_parquet(
                        part_path,
                        index=False,
                        engine="pyarrow",
                        compression=compression,
                    )
                    total_rows += len(part)
                    if i < 3:
                        tqdm.write(f"  wrote {part_path} ({len(part)} rows)")
                    del part
                finally:
                    stop_hb.set()
                    hb.join(timeout=2.0)

                pbar.update(1)

        try:
            os.remove(progress_marker)
        except OSError:
            pass
        with open(complete_path, "w", encoding="utf-8") as cf:
            cf.write(f"rows={total_rows} cols={n_cols}\n")

        return total_rows, n_cols

    except BaseException as exc:
        with open(failed_path, "w", encoding="utf-8") as ff:
            ff.write(
                "The merge stopped before completion (crash, Ctrl+C, or error).\n"
                "Any part_*.parquet files here may be an incomplete run — delete them "
                "before restarting if you want a clean output.\n\n"
            )
            ff.write(f"{type(exc).__name__}: {exc}\n\n")
            ff.write(traceback.format_exc())
        try:
            os.remove(progress_marker)
        except OSError:
            pass
        tqdm.write(f"Merge stopped: {type(exc).__name__}: {exc}. Details: {failed_path}")
        raise


def save_csv_with_progress(
    df: pd.DataFrame,
    path: str,
    chunksize: int = 2000,
) -> None:
    """Write CSV in row chunks with a tqdm progress bar."""

    n = len(df)
    if n == 0:
        df.to_csv(path, index=False)
        return

    n_steps = (n + chunksize - 1) // chunksize
    first = True
    with open(path, "w", newline="", encoding="utf-8") as f:
        for start in tqdm(
            range(0, n, chunksize),
            total=n_steps,
            desc="Saving CSV",
            unit="chunk",
        ):
            chunk = df.iloc[start : start + chunksize]
            chunk.to_csv(f, index=False, header=first)
            first = False