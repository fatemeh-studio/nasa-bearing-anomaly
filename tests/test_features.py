"""
test_features.py
Tests for the physics-based feature extraction pipeline.
Run: pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from nasa_bearing_anomaly.features import (
    SUMMARY_DUPLICATE_KEYS,
    FeatureExtractor,
    add_change_rate_features,
    add_rolling_features,
)
from nasa_bearing_anomaly.loading import _kurtosis, _skewness

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


# ── Band Resolution ────────────────────────────────────────────────────────
# A band narrower than the PSD bin spacing is a point sample, not an integral.
# Measured 2026-08-15: at the default window every defect harmonic resolved to
# exactly one bin, which made bandwidth_hz inert below 10 Hz and left ftf_energy
# with a healthy-to-faulty separation of 0.02 on real Test 3 data.


class TestBandResolution:
    def test_halfwidth_is_derived_from_resolution_not_hardcoded(self):
        """A longer Welch window gives finer bins, so the band must narrow with it."""
        coarse = FeatureExtractor(fs=FS, window_sec=0.1)
        fine = FeatureExtractor(fs=FS, window_sec=0.4)
        assert fine.freq_resolution_hz < coarse.freq_resolution_hz
        assert fine.band_halfwidth_hz < coarse.band_halfwidth_hz

    def test_default_band_spans_at_least_three_bins(self, extractor):
        """Every emitted band must be an integral over >= 3 bins, at every harmonic."""
        freqs, _ = extractor.welch_psd(np.zeros(20480))
        half = extractor.band_halfwidth_hz
        for name in extractor.band_energy_freqs:
            f0 = extractor.defect_freqs[name]
            for h in range(1, extractor.n_harmonics + 1):
                centre = f0 * h
                n_bins = int(((freqs >= centre - half) & (freqs <= centre + half)).sum())
                assert n_bins >= 3, f"{name} harmonic {h} spans only {n_bins} bin(s)"

    def test_ftf_band_reaches_dc_and_is_excluded(self, extractor):
        """
        FTF is 14.775 Hz. At a 10 Hz bin spacing no band around it can span three
        bins without reaching DC, so it earns no feature. The exclusion is derived
        from the resolution rule, not hardcoded -- this pins the reason, so that a
        finer window would let FTF back in on its own.
        """
        assert not extractor.band_is_resolvable(extractor.defect_freqs["FTF_Hz"])
        assert "FTF_Hz" not in extractor.band_energy_freqs
        assert "ftf_energy" not in extractor.extract_frequency_domain(np.zeros(20480))

    def test_the_other_three_are_resolvable(self, extractor):
        for name in ("BPFO_Hz", "BPFI_Hz", "BSF_Hz"):
            assert extractor.band_is_resolvable(extractor.defect_freqs[name]), name
            assert name in extractor.band_energy_freqs


# ── Spectral Table Contract ────────────────────────────────────────────────
# test{N}_spectral.csv joins test{N}_raw.csv on file_index, so what it may and
# may not carry is a contract rather than a preference.


class TestSpectralTableContract:
    PREFIX = "Bearing3_ch1"

    @pytest.fixture
    def kept_names(self, extractor, healthy_signal):
        """Names the spectral table keeps, channel prefix stripped."""
        keys = extractor.extract_from_raw(healthy_signal, channel_name=self.PREFIX)
        n = len(self.PREFIX) + 1
        return {k[n:] for k in keys if k[n:] not in SUMMARY_DUPLICATE_KEYS}

    def test_no_name_collides_with_the_summary_table(self, kept_names):
        """A shared name would put two columns in the joined frame for one measurement."""
        for stat in ("rms", "mean", "std", "max", "kurt", "skew"):
            assert stat not in kept_names

    def test_dropped_keys_really_duplicate_the_loader(self, extractor, healthy_signal):
        """
        Dropping a key is only safe if loading.py already writes that quantity.

        Two of the five differ in name from the loader's (`kurtosis` against
        `_kurt`, `peak` against `_max`), so name equality is not enough -- the
        values have to match, or the drop loses a feature rather than a duplicate.
        """
        x = healthy_signal
        expected = {
            "rms": np.sqrt(np.mean(x**2)),
            "std": np.std(x),
            "kurtosis": _kurtosis(x),
            "peak": np.max(np.abs(x)),
            "skewness": _skewness(x),
        }
        assert set(expected) == set(SUMMARY_DUPLICATE_KEYS)
        computed = extractor.extract_time_domain(x)
        for name, value in expected.items():
            assert computed[name] == pytest.approx(value), name

    def test_columns_caught_by_the_time_domain_filter_are_known(self, kept_names):
        """
        select_features matches `_rms`/`_kurt`/`_std`/`_max` as substrings rather
        than suffixes, so `psd_std` and `env_kurtosis` read as time-domain columns
        to it despite being frequency- and envelope-domain. Both must be kept out
        of a time-domain arm by explicit selection. Pinned here so that a feature
        added later which collides goes red, instead of quietly widening an arm it
        does not belong to -- select_features degrades to a smaller set rather
        than raising, so nothing else would notice.
        """
        caught = {
            name
            for name in kept_names
            if any(kw in f"{self.PREFIX}_{name}" for kw in ("_rms", "_kurt", "_std", "_max"))
        }
        assert caught == {"psd_std", "env_kurtosis"}


# ── Envelope Features ──────────────────────────────────────────────────────
# A bearing defect excites a high-frequency structural resonance, amplitude-
# modulated at the defect rate. The defect rate is therefore carried by the
# envelope of the resonance, not by a line in the raw spectrum.


class TestEnvelopeFeatures:
    @pytest.fixture
    def modulated_signal(self):
        """A 5 kHz carrier amplitude-modulated at 236 Hz -- the textbook fault model."""
        t = np.linspace(0, 1, FS, endpoint=False)
        carrier = np.sin(2 * np.pi * 5000 * t)
        return (1.0 + 0.8 * np.sin(2 * np.pi * 236.0 * t)) * carrier

    def test_modulation_is_absent_from_the_raw_spectrum(self, extractor, modulated_signal):
        """
        The premise for enveloping at all: amplitude modulation puts no energy at
        the modulating frequency. It appears as sidebands either side of the
        carrier, so a band-energy feature read off the raw PSD sees nothing at BPFO.
        """
        freqs, psd = extractor.welch_psd(modulated_signal)
        low = psd[freqs < 1000]
        assert psd[freqs < 1000].max() < 0.01 * psd.max(), (
            "the raw spectrum should carry no low-frequency line"
        )
        assert freqs[np.argmax(psd)] > 4000, "raw spectrum should peak at the carrier"
        assert low.size > 0

    def test_envelope_recovers_the_modulation_frequency(self, extractor, modulated_signal):
        """And this is what enveloping buys: the same 236 Hz becomes the dominant line."""
        env = extractor.envelope(modulated_signal)
        freqs, psd = extractor.envelope_spectrum(env)
        peak_hz = freqs[np.argmax(psd)]
        assert abs(peak_hz - 236.0) <= extractor.band_halfwidth_hz, (
            f"envelope spectrum peaked at {peak_hz} Hz, expected ~236 Hz"
        )

    def test_envelope_of_a_constant_amplitude_tone_is_flat(self, extractor):
        """No modulation, so no envelope variation. Edges are trimmed: filtfilt rings."""
        t = np.linspace(0, 1, FS, endpoint=False)
        env = extractor.envelope(np.sin(2 * np.pi * 5000 * t))
        core = env[FS // 10 : -FS // 10]
        assert core.std() / core.mean() < 0.1

    def test_envelope_is_non_negative(self, extractor, faulty_signal):
        assert (extractor.envelope(faulty_signal) >= 0).all()

    def test_env_bpfo_energy_higher_in_faulty(self, extractor, healthy_signal, faulty_signal):
        h = extractor.extract_envelope_domain(healthy_signal)
        f = extractor.extract_envelope_domain(faulty_signal)
        assert f["env_bpfo_energy"] > h["env_bpfo_energy"]

    def test_extract_envelope_domain_no_nan(self, extractor, healthy_signal):
        for k, v in extractor.extract_envelope_domain(healthy_signal).items():
            assert np.isfinite(v), f"Feature '{k}' is not finite: {v}"

    def test_envelope_keys_track_the_resolvable_frequencies(self, extractor, healthy_signal):
        """Same resolution rule as the raw-PSD bands, so FTF is absent here too."""
        keys = extractor.extract_envelope_domain(healthy_signal)
        assert "env_ftf_energy" not in keys
        for name in extractor.band_energy_freqs:
            assert f"env_{name.replace('_Hz', '').lower()}_energy" in keys


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
