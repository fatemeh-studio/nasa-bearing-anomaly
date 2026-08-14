"""
test_loading.py
Tests for the raw-file loading and per-file summary statistics.
Run: pytest tests/test_loading.py -v
"""

import numpy as np
import pandas as pd
import pytest

from nasa_bearing_anomaly.config import TEST_CONFIG
from nasa_bearing_anomaly.loading import _kurtosis, _skewness, load_test


def _write_raw_files(directory, n_channels, filenames):
    """Write space-separated text files shaped like NASA acquisition files."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name in filenames:
        np.savetxt(directory / name, rng.normal(size=(64, n_channels)))
    return directory


class TestStatisticalFeatures:
    """Test kurtosis and skewness calculations."""

    def test_gaussian_kurtosis_near_zero(self):
        """Excess kurtosis of Gaussian noise should be ≈ 0."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100000)
        k = _kurtosis(x)
        assert abs(k) < 0.1, f"Gaussian excess kurtosis should be ≈ 0, got {k:.4f}"

    def test_impulsive_signal_high_kurtosis(self):
        """Signal with rare large spikes should have high kurtosis > 3."""
        np.random.seed(42)
        x = np.random.normal(0, 0.1, 10000)
        # Add rare large impulses (simulating bearing defect)
        x[::100] += np.random.choice([5, -5], size=len(x[::100]))
        k = _kurtosis(x)
        assert k > 3, f"Impulsive signal should have kurtosis > 3, got {k:.4f}"

    def test_symmetric_signal_zero_skewness(self):
        """Symmetric distribution should have near-zero skewness."""
        x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
        s = _skewness(x)
        assert abs(s) < 1e-10, f"Symmetric signal skewness should be 0, got {s}"

    def test_constant_signal_safe(self):
        """Constant signal (zero std) should not cause division by zero."""
        x = np.ones(100)
        assert _kurtosis(x) == 0.0
        assert _skewness(x) == 0.0


class TestChannelLabelling:
    """
    load_test labels physical channel j with TEST_CONFIG's columns[j]. When the
    list was longer than the data, the surplus names were dropped and every
    channel was mislabelled without an error: Test 3's outer-race failure was
    stored under Bearing2_ch1 while failed_bearing read Bearing3, and the whole
    pipeline ran to completion on it.
    """

    def test_more_column_names_than_channels_raises(self, tmp_path, monkeypatch):
        """Eight declared names against four-channel data must fail, not truncate."""
        raw_dir = _write_raw_files(
            tmp_path / "raw", 4, ["2004.01.01.00.00.00", "2004.01.01.00.10.00"]
        )
        monkeypatch.setitem(TEST_CONFIG[1], "raw_dir", raw_dir)
        monkeypatch.setitem(TEST_CONFIG[1], "total_files", 2)

        with pytest.raises(ValueError) as excinfo:
            load_test(1, verbose=False)

        message = str(excinfo.value)
        assert "8 column names" in message
        assert "4 channels" in message

    def test_channel_n_is_bearing_n(self, tmp_path, monkeypatch):
        """
        Four channels against four names must label Bearing 1 to 4 in order.
        This is the contract the guard protects: in tests 2 and 3 there is one
        accelerometer per bearing, so channel N is Bearing N.
        """
        raw_dir = _write_raw_files(
            tmp_path / "raw", 4, ["2004.01.01.00.00.00", "2004.01.01.00.10.00"]
        )
        monkeypatch.setitem(TEST_CONFIG[3], "raw_dir", raw_dir)
        monkeypatch.setitem(TEST_CONFIG[3], "total_files", 2)

        df = load_test(3, verbose=False)

        assert [c for c in df.columns if c.endswith("_rms")] == [
            "Bearing1_ch1_rms",
            "Bearing2_ch1_rms",
            "Bearing3_ch1_rms",
            "Bearing4_ch1_rms",
        ]


class TestTimestampParsing:
    """
    Filenames carry their own acquisition time as YYYY.MM.DD.HH.MM.SS. Parsing
    them with Path.stem silently dropped the seconds field as a suffix, leaving
    five fields where six were indexed; the IndexError was swallowed into NaT,
    so every row in every table had an empty timestamp and nothing complained.
    """

    def test_filename_parses_to_its_acquisition_time(self, tmp_path, monkeypatch):
        """A filename in the documented format yields exactly that timestamp."""
        raw_dir = _write_raw_files(
            tmp_path / "raw", 4, ["2004.02.16.03.02.39", "2004.02.16.03.12.39"]
        )
        monkeypatch.setitem(TEST_CONFIG[3], "raw_dir", raw_dir)
        monkeypatch.setitem(TEST_CONFIG[3], "total_files", 2)

        df = load_test(3, verbose=False)

        assert list(df["timestamp"]) == [
            pd.Timestamp("2004-02-16 03:02:39"),
            pd.Timestamp("2004-02-16 03:12:39"),
        ]

    def test_unparseable_filename_gives_nat_without_raising(self, tmp_path, monkeypatch):
        """
        A filename that is not a timestamp must not abort the run -- the sample
        is still usable, it just has no clock time.
        """
        raw_dir = _write_raw_files(tmp_path / "raw", 4, ["not-a-timestamp", "also.not.one"])
        monkeypatch.setitem(TEST_CONFIG[3], "raw_dir", raw_dir)
        monkeypatch.setitem(TEST_CONFIG[3], "total_files", 2)

        df = load_test(3, verbose=False)

        assert len(df) == 2
        assert df["timestamp"].isna().all()


class TestCommittedTableSchema:
    """
    The tables in data/processed/ are committed so that notebooks 02-04 run from
    a clone without the 6.2 GB raw download, which makes their shape part of the
    repository's contract rather than an implementation detail. These read
    headers and one column, not values: what the series *contains* is a finding
    and belongs in the README, but what columns exist is a contract.
    """

    @pytest.mark.parametrize("test_id", [2, 3])
    def test_one_channel_per_bearing(self, test_id):
        """
        Tests 2 and 3 recorded one accelerometer per bearing, so a _ch2 column
        for either means the eight-name column list was used again.
        """
        columns = pd.read_csv(TEST_CONFIG[test_id]["output_file"], nrows=0).columns

        assert not [c for c in columns if "_ch2" in c]
        for bearing in range(1, 5):
            assert f"Bearing{bearing}_ch1_rms" in columns

    def test_test1_keeps_both_accelerometers(self):
        """Test 1 has two accelerometers per bearing and must keep all eight channels."""
        columns = pd.read_csv(TEST_CONFIG[1]["output_file"], nrows=0).columns

        for bearing in range(1, 5):
            assert f"Bearing{bearing}_ch1_rms" in columns
            assert f"Bearing{bearing}_ch2_rms" in columns

    @pytest.mark.parametrize("test_id", [1, 2, 3])
    def test_timestamps_are_populated(self, test_id):
        """
        Every timestamp in every committed table was NaT before 2026-08-12. An
        empty column reads as ordinary missing data, so nothing downstream
        objected; only a lead time computed in hours would ever have noticed.
        """
        stamps = pd.read_csv(TEST_CONFIG[test_id]["output_file"], usecols=["timestamp"])

        assert stamps["timestamp"].notna().all()
