"""
loading.py
----------
Reads the NASA IMS raw acquisition files and collapses each one to a row of
per-channel summary statistics.

A single test run is tens of thousands of waveforms; keeping every sample in
memory is not practical and is not what the degradation trend needs. Each file
becomes one row -- six statistics per channel -- indexed by file_index, which
preserves the temporal ordering that the whole analysis is built on.

Raw filenames encode their own timestamp as YYYY.MM.DD.HH.MM.SS, which is what
the timestamp column is parsed from. Dataset shape, channel counts per test and
known defects are documented in data/README.md.

Usage:
    python -m nasa_bearing_anomaly.loading --test 1
    python -m nasa_bearing_anomaly.loading --test all
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BEARING_PARAMS, TEST_CONFIG
from .physics import compute_defect_frequencies

try:
    from tqdm import tqdm
except ImportError:
    # tqdm ships in the [notebook] extra. Without it the load runs silently
    # rather than failing.
    def tqdm(iterable, **kwargs):  # noqa: F811 — silent fallback
        return iterable


# ─── Loading Functions ─────────────────────────────────────────────────────


def load_single_file(filepath: Path) -> np.ndarray:
    """
    Load a single NASA acquisition file.
    Files are space-separated text despite carrying no extension.

    Returns ndarray of shape (n_samples, n_channels).
    """
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def load_test(test_id: int, max_files: int = None, verbose: bool = True) -> pd.DataFrame:
    """
    Load all files from a single test run and aggregate statistics per file.

    Instead of storing all 20,480 × N_files raw samples (too large),
    we compute per-file statistical summaries which preserve the temporal trend.

    Parameters
    ----------
    test_id : int
        1, 2, or 3
    max_files : int, optional
        Limit number of files (useful for quick testing)
    verbose : bool
        Show progress bar

    Returns
    -------
    pd.DataFrame
        One row per file (time step), columns = statistical features per channel.
        Index = file_index (0, 1, 2, ...)
    """
    config = TEST_CONFIG[test_id]
    raw_dir = config["raw_dir"]
    columns = config["columns"]

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            f"Please download the NASA IMS Bearing Dataset and place files in:\n"
            f"  data/raw/test{test_id}/"
        )

    files = sorted(raw_dir.iterdir())
    files = [f for f in files if f.is_file() and not f.name.startswith(".")]

    if not files:
        raise FileNotFoundError(f"No acquisition files found in {raw_dir}")

    # A file count that disagrees with the documented one usually means the wrong
    # folder was unpacked. For test 3 specifically, pointing at 3rd_test/ instead
    # of the nested 3rd_test/4th_test/txt/ yields 4,448 files, and the outer-race
    # failure this project detects lives entirely beyond that cutoff -- the run
    # then looks healthy from start to finish. See data/README.md.
    expected_files = config["total_files"]
    if len(files) != expected_files:
        print(
            f"Warning: {raw_dir} holds {len(files)} files, not the {expected_files} "
            f"documented for test {test_id}. See data/README.md."
        )

    if max_files:
        files = files[:max_files]

    # Checked once, here, rather than per file: the per-file body below is wrapped
    # in `except Exception: continue`, which would turn a raise into a skip message
    # and let the run finish on mislabelled data.
    n_channels = load_single_file(files[0]).shape[1]
    if n_channels != len(columns):
        raise ValueError(
            f"Test {test_id}: TEST_CONFIG lists {len(columns)} column names but the raw "
            f"files have {n_channels} channels. Channel j is labelled columns[j], so a "
            f"mismatch mislabels every channel instead of failing. Per-test channel "
            f"counts are in data/README.md."
        )

    if verbose:
        print(f"\n📂 Loading Test {test_id}: {len(files)} files from {raw_dir}")
        print(f"   Failed bearing: {config['failed_bearing']} ({config['failure_mode']})")

    records = []
    for i, filepath in enumerate(tqdm(files, desc=f"Test {test_id}", disable=not verbose)):
        try:
            raw = load_single_file(filepath)

            # Parse timestamp from filename (format: YYYY.MM.DD.HH.MM.SS).
            # filepath.name, not .stem: pathlib treats the seconds field as a
            # suffix, so .stem returns five fields where six are needed and every
            # timestamp silently becomes NaT.
            try:
                parts = filepath.name.split(".")
                timestamp = pd.Timestamp(
                    year=int(parts[0]),
                    month=int(parts[1]),
                    day=int(parts[2]),
                    hour=int(parts[3]),
                    minute=int(parts[4]),
                    second=int(parts[5]),
                )
            except Exception:
                timestamp = pd.NaT

            record = {
                "file_index": i,
                "timestamp": timestamp,
                "filename": filepath.name,
            }

            # No per-channel length check: the guard above proved columns and
            # channels agree before the loop started.
            for j, col in enumerate(columns):
                ch_data = raw[:, j]
                record[f"{col}_rms"] = np.sqrt(np.mean(ch_data**2))
                record[f"{col}_mean"] = np.mean(ch_data)
                record[f"{col}_std"] = np.std(ch_data)
                record[f"{col}_max"] = np.max(np.abs(ch_data))
                record[f"{col}_kurt"] = _kurtosis(ch_data)
                record[f"{col}_skew"] = _skewness(ch_data)

            records.append(record)

        except Exception as e:
            if verbose:
                print(f"  ⚠ Skipped {filepath.name}: {e}")
            continue

    # Skips are reported per file above, but thousands of lines of progress bar
    # make a handful of them easy to miss. One line at the end is not.
    if len(records) != len(files):
        print(f"Warning: {len(files) - len(records)} of {len(files)} files were skipped.")

    # A failed timestamp parse produces NaT rather than an error, so without this
    # the column can be empty for every row and still look like ordinary missing
    # data. Any downstream figure in real time depends on it.
    n_undated = sum(1 for r in records if pd.isna(r["timestamp"]))
    if n_undated:
        print(f"Warning: {n_undated} of {len(records)} filenames did not parse as timestamps.")

    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df = df.set_index("file_index")

    return df


def save_processed(df: pd.DataFrame, test_id: int) -> Path:
    """Save the processed DataFrame to CSV."""
    output_path = TEST_CONFIG[test_id]["output_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"✅ Saved: {output_path}  ({len(df)} rows, {len(df.columns)} columns)")
    return output_path


def load_processed(test_id: int) -> pd.DataFrame:
    """Load a previously processed CSV."""
    path = TEST_CONFIG[test_id]["output_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}\n"
            f"Run: python -m nasa_bearing_anomaly.loading --test {test_id}"
        )
    return pd.read_csv(path, index_col="file_index")


# ─── Helper Math ───────────────────────────────────────────────────────────


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher). Gaussian noise reads 0, so a healthy bearing does too.

    Published bearing thresholds are usually non-excess — healthy 3, fault above 5 —
    and are 3 too high for this value. Subtract 3 before comparing.
    """
    n = len(x)
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        return 0.0
    return (np.sum((x - mu) ** 4) / n) / sigma**4 - 3.0


def _skewness(x: np.ndarray) -> float:
    n = len(x)
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        return 0.0
    return (np.sum((x - mu) ** 3) / n) / sigma**3


# ─── CLI Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load NASA IMS Bearing Dataset")
    parser.add_argument("--test", type=str, default="all", help="Test to load: 1, 2, 3, or 'all'")
    parser.add_argument(
        "--max_files", type=int, default=None, help="Max files to load per test (for quick testing)"
    )
    args = parser.parse_args()

    tests = [1, 2, 3] if args.test == "all" else [int(args.test)]

    freqs = compute_defect_frequencies(BEARING_PARAMS)
    print("\n⚙️  Bearing Defect Frequencies (IMS Test Rig):")
    for name, hz in freqs.items():
        print(f"   {name}: {hz} Hz")

    for t in tests:
        df = load_test(t, max_files=args.max_files)

        # A truncated run must not overwrite the committed table. --max_files
        # exists to prove the loader works without reading the full archive, and
        # saving 50 rows over 6,324 would destroy the file it was checking.
        if args.max_files:
            print(
                f"Smoke test only: {len(df)} rows, {len(df.columns)} columns. "
                f"Not saved -- rerun without --max_files to write the table."
            )
        else:
            save_processed(df, t)
