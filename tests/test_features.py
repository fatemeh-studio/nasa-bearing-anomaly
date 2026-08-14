"""
test_features.py
Tests for the physics-based feature extraction pipeline.
Run: pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from nasa_bearing_anomaly.features import (
    FeatureExtractor,
    add_change_rate_features,
    add_rolling_features,
)

FS = 20000  # Hz


@pytest.fixture
def extractor():
    return FeatureExtractor(fs=FS)


@pytest.fixture
def healthy_signal():
    """Gaussian noise + low-amplitude sinusoid — simulates a healthy bearing."""
    np.random.seed(0)
    t = np.linspace(0, 1, FS)
    return np.random.normal(0, 0.1, FS) + 0.05 * np.sin(2 * np.pi * 33.33 * t)


@pytest.fixture
def faulty_signal(healthy_signal):
    """Healthy signal + periodic impulses at BPFO — simulates outer race fault."""
    sig = healthy_signal.copy()
    bpfo = 236.0
    for t_imp in np.arange(0, 1, 1 / bpfo):
        idx = int(t_imp * FS)
        if idx < FS - 80:
            sig[idx : idx + 50] += np.exp(-np.linspace(0, 5, 50)) * 1.5
    return sig


# ── Time-Domain Features ───────────────────────────────────────────────────


class TestTimeDomainFeatures:
    def test_rms_positive(self, extractor, healthy_signal):
        assert extractor.rms(healthy_signal) > 0

    def test_rms_increases_with_amplitude(self, extractor):
        low = np.random.normal(0, 0.1, 1000)
        high = np.random.normal(0, 1.0, 1000)
        assert extractor.rms(high) > extractor.rms(low)

    def test_faulty_kurtosis_higher_than_healthy(self, extractor, healthy_signal, faulty_signal):
        k_healthy = extractor.kurtosis(healthy_signal)
        k_faulty = extractor.kurtosis(faulty_signal)
        assert k_faulty > k_healthy, (
            f"Faulty kurtosis ({k_faulty:.2f}) should exceed healthy ({k_healthy:.2f})"
        )

    def test_healthy_kurtosis_near_gaussian(self, extractor):
        """With enough samples, Gaussian noise gives excess kurtosis ≈ 0."""
        np.random.seed(7)
        x = np.random.normal(0, 1, 50000)
        k = extractor.kurtosis(x)
        assert abs(k) < 0.2, f"Expected kurtosis ≈ 0, got {k:.4f}"

    def test_crest_factor_always_gte_one(self, extractor, healthy_signal, faulty_signal):
        """Crest factor = peak/RMS must always be ≥ 1 by definition."""
        assert extractor.crest_factor(healthy_signal) >= 1.0
        assert extractor.crest_factor(faulty_signal) >= 1.0

    def test_extract_time_domain_returns_all_keys(self, extractor, healthy_signal):
        expected_keys = {
            "rms",
            "peak",
            "kurtosis",
            "skewness",
            "crest_factor",
            "shape_factor",
            "impulse_factor",
            "clearance_factor",
            "energy",
            "std",
            "mean_abs",
        }
        result = extractor.extract_time_domain(healthy_signal)
        assert expected_keys == set(result.keys())

    def test_no_nan_in_time_features(self, extractor, healthy_signal):
        result = extractor.extract_time_domain(healthy_signal)
        for k, v in result.items():
            assert np.isfinite(v), f"Feature '{k}' is not finite: {v}"

    def test_zero_signal_safe(self, extractor):
        """All-zero signal must not raise exceptions."""
        x = np.zeros(FS)
        result = extractor.extract_time_domain(x)
        assert isinstance(result, dict)


# ── Frequency-Domain Features ──────────────────────────────────────────────


class TestFrequencyDomainFeatures:
    def test_bpfo_energy_higher_in_faulty(self, extractor, healthy_signal, faulty_signal):
        """Faulty signal should have more energy at BPFO harmonics."""
        h_feat = extractor.extract_frequency_domain(healthy_signal)
        f_feat = extractor.extract_frequency_domain(faulty_signal)
        assert f_feat["bpfo_energy"] > h_feat["bpfo_energy"], (
            "BPFO energy should be higher in faulty signal"
        )

    def test_spectral_entropy_in_range(self, extractor, healthy_signal):
        """Spectral entropy must be in [0, 1] after normalization."""
        feat = extractor.extract_frequency_domain(healthy_signal)
        assert 0.0 <= feat["spectral_entropy"] <= 1.0

    def test_spectral_entropy_faulty_lower_than_healthy(
        self, extractor, healthy_signal, faulty_signal
    ):
        """
        Gaussian noise has a flat spectrum — maximally entropic (high entropy).
        Adding BPFO impulses concentrates energy at harmonics — lower entropy.
        So healthy > faulty, not the other way around.
        """
        h_se = extractor.extract_frequency_domain(healthy_signal)["spectral_entropy"]
        f_se = extractor.extract_frequency_domain(faulty_signal)["spectral_entropy"]
        assert h_se > f_se

    def test_high_freq_ratio_in_valid_range(self, extractor, healthy_signal):
        """high_freq_ratio must be a valid probability in [0, 1]."""
        feat = extractor.extract_frequency_domain(healthy_signal)
        assert 0.0 <= feat["high_freq_ratio"] <= 1.0

    def test_dominant_freq_is_positive(self, extractor, healthy_signal):
        feat = extractor.extract_frequency_domain(healthy_signal)
        assert feat["dominant_freq_hz"] >= 0

    def test_extract_frequency_domain_no_nan(self, extractor, healthy_signal):
        result = extractor.extract_frequency_domain(healthy_signal)
        for k, v in result.items():
            assert np.isfinite(v), f"Feature '{k}' is not finite: {v}"

    def test_extract_from_raw_channel_prefix(self, extractor, healthy_signal):
        """extract_from_raw should prefix all keys with the channel name."""
        result = extractor.extract_from_raw(healthy_signal, channel_name="Bearing3_ch1")
        for key in result:
            assert key.startswith("Bearing3_ch1_"), f"Key missing prefix: {key}"


# ── Rolling & Change-Rate Features ────────────────────────────────────────


class TestRollingFeatures:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 200
        # Simulate gradual degradation: RMS rises from 0.1 to 1.0
        rms_vals = np.linspace(0.1, 1.0, n) + np.random.normal(0, 0.02, n)
        # Excess kurtosis, so a healthy bearing sits near 0, not 3.
        kurt_vals = np.ones(n) * 0.15 + np.random.normal(0, 0.1, n)
        return pd.DataFrame(
            {
                "Bearing1_ch1_rms": rms_vals,
                "Bearing1_ch1_kurt": kurt_vals,
            }
        )

    def test_rolling_columns_added(self, sample_df):
        cols = ["Bearing1_ch1_rms"]
        result = add_rolling_features(sample_df, cols, windows=[10, 50])
        assert "Bearing1_ch1_rms_roll_mean_10" in result.columns
        assert "Bearing1_ch1_rms_roll_std_10" in result.columns
        assert "Bearing1_ch1_rms_roll_mean_50" in result.columns

    def test_rolling_does_not_modify_original(self, sample_df):
        original_cols = list(sample_df.columns)
        _ = add_rolling_features(sample_df, ["Bearing1_ch1_rms"], windows=[10])
        assert list(sample_df.columns) == original_cols

    def test_rolling_mean_smoother_than_raw(self, sample_df):
        result = add_rolling_features(sample_df, ["Bearing1_ch1_rms"], windows=[10])
        raw_std = sample_df["Bearing1_ch1_rms"].std()
        roll_std = result["Bearing1_ch1_rms_roll_mean_10"].std()
        assert roll_std < raw_std, "Rolling mean should be smoother than raw signal"

    def test_change_rate_columns_added(self, sample_df):
        result = add_change_rate_features(sample_df, ["Bearing1_ch1_rms"])
        assert "Bearing1_ch1_rms_diff" in result.columns
        assert "Bearing1_ch1_rms_diff_abs" in result.columns

    def test_change_rate_first_value_zero(self, sample_df):
        """First diff value should be 0 (filled from NaN)."""
        result = add_change_rate_features(sample_df, ["Bearing1_ch1_rms"])
        assert result["Bearing1_ch1_rms_diff"].iloc[0] == 0.0

    def test_nonexistent_column_skipped_gracefully(self, sample_df):
        """Passing a non-existent column should not raise an exception."""
        result = add_rolling_features(sample_df, ["nonexistent_col"], windows=[10])
        assert "nonexistent_col_roll_mean_10" not in result.columns
