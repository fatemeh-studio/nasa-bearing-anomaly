"""
Physics-based feature extraction for bearing health monitoring.

Two levels of features:

1. TIME-DOMAIN FEATURES (fast, interpretable)
    - RMS: root mean square — direct measure of vibration energy
    - Kurtosis: sensitivity to impulsive spikes (key fault indicator)
    - Crest Factor: peak / RMS (rises when impulses appear)
    - Shape Factor, Impulse Factor, Clearance Factor

2. FREQUENCY-DOMAIN FEATURES (requires raw waveform)
    - Power Spectral Density via Welch method
    - Band energy at BPFO, BPFI, BSF and FTF harmonics, measured on the PSD
    - Spectral Entropy (order → chaos)
    - High-frequency energy ratio (energy above 5 kHz)

The combination of both makes this project stand out from
basic Isolation Forest tutorials. This is what physicists do.

Usage:
    from nasa_bearing_anomaly.features import FeatureExtractor
    extractor = FeatureExtractor()
    features = extractor.extract_from_raw(waveform_array)
"""

import argparse

import numpy as np
import pandas as pd
from scipy import signal

from .config import BEARING_PARAMS, SAMPLING_RATE_HZ, TEST_CONFIG
from .loading import load_processed
from .physics import compute_defect_frequencies

# ─── Feature Extractor ─────────────────────────────────────────────────────


class FeatureExtractor:
    """
    Extracts physics-informed features from raw vibration waveforms.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz (defaults to the dataset's nominal rate)
    window_sec : float
        Window length in seconds for Welch PSD (default: 0.1)
    bearing_params : dict
        Physical bearing parameters for defect frequency computation
    """

    def __init__(
        self,
        fs: int = SAMPLING_RATE_HZ,
        window_sec: float = 0.1,
        bearing_params: dict = BEARING_PARAMS,
    ):
        self.fs = fs
        self.nperseg = int(window_sec * fs)
        self.defect_freqs = compute_defect_frequencies(bearing_params)

        # Harmonic orders to check for each defect frequency
        self.n_harmonics = 3

    # ── Time-Domain Features ───────────────────────────────────────────────

    def rms(self, x: np.ndarray) -> float:
        """Root Mean Square — overall vibration energy level."""
        return float(np.sqrt(np.mean(x**2)))

    def peak(self, x: np.ndarray) -> float:
        """Peak amplitude."""
        return float(np.max(np.abs(x)))

    def kurtosis(self, x: np.ndarray) -> float:
        """
        Excess kurtosis (Fisher), so Gaussian noise reads 0.

        A healthy bearing reads near it: measured healthy median on Test 3,
        Bearing 3 is 0.15.

        Early fault: ~2. Severe fault: above ~7. These are the usual published
        thresholds (3, 5, 10) minus 3, because this value already subtracts the
        Gaussian baseline — applying the published numbers directly puts every
        threshold 3 too high.

        The standard indicator for impulsive faults: a spalled race produces one
        impact per rolling element passing the defect, and kurtosis weights those
        tails heavily.
        """
        mu = np.mean(x)
        sigma = np.std(x)
        if sigma < 1e-12:
            return 0.0
        return float(np.mean((x - mu) ** 4) / sigma**4 - 3.0)

    def skewness(self, x: np.ndarray) -> float:
        """
        Third standardised moment: asymmetry of the amplitude distribution.

        Returns 0.0 for a constant signal rather than dividing by zero.

        Parameters
        ----------
        x : numpy.ndarray
            Raw 1-D waveform.

        Returns
        -------
        float
            Skewness of ``x``.
        """
        mu = np.mean(x)
        sigma = np.std(x)
        if sigma < 1e-12:
            return 0.0
        return float(np.mean((x - mu) ** 3) / sigma**3)

    def crest_factor(self, x: np.ndarray) -> float:
        """
        Crest Factor = Peak / RMS.

        Sensitive to early-stage single-point defects. Its direction becomes
        unreliable at high fault severity, once damage spreads and the signal
        stops being impulsive, so it is read alongside kurtosis rather than alone.
        """
        r = self.rms(x)
        return self.peak(x) / r if r > 1e-12 else 0.0

    def shape_factor(self, x: np.ndarray) -> float:
        """RMS / Mean Absolute Value."""
        ma = np.mean(np.abs(x))
        return self.rms(x) / ma if ma > 1e-12 else 0.0

    def impulse_factor(self, x: np.ndarray) -> float:
        """Peak / Mean Absolute Value."""
        ma = np.mean(np.abs(x))
        return self.peak(x) / ma if ma > 1e-12 else 0.0

    def clearance_factor(self, x: np.ndarray) -> float:
        """Peak / (Mean Square Root Amplitude)²."""
        msra = np.mean(np.sqrt(np.abs(x)))
        return self.peak(x) / (msra**2) if msra > 1e-12 else 0.0

    def energy(self, x: np.ndarray) -> float:
        """Total signal energy."""
        return float(np.sum(x**2))

    def extract_time_domain(self, x: np.ndarray) -> dict:
        """Compute all time-domain features for one channel waveform."""
        return {
            "rms": self.rms(x),
            "peak": self.peak(x),
            "kurtosis": self.kurtosis(x),
            "skewness": self.skewness(x),
            "crest_factor": self.crest_factor(x),
            "shape_factor": self.shape_factor(x),
            "impulse_factor": self.impulse_factor(x),
            "clearance_factor": self.clearance_factor(x),
            "energy": self.energy(x),
            "std": float(np.std(x)),
            "mean_abs": float(np.mean(np.abs(x))),
        }

    # ── Frequency-Domain Features ──────────────────────────────────────────

    def welch_psd(self, x: np.ndarray):
        """
        Welch power spectral density (PSD) estimate.

        More stable than a raw fast Fourier transform (FFT) for noisy industrial
        signals, because averaging over overlapping segments trades frequency
        resolution for reduced variance.

        Parameters
        ----------
        x : numpy.ndarray
            Raw 1-D waveform.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            Frequencies in Hz, and the power spectral density at each.
        """
        freqs, psd = signal.welch(
            x,
            fs=self.fs,
            nperseg=self.nperseg,
            window="hann",
            average="mean",
        )
        return freqs, psd

    def spectral_entropy(self, psd: np.ndarray) -> float:
        """
        Compute normalized spectral entropy of a power spectral density.

        High entropy → flat/broadband spectrum (Gaussian noise, healthy baseline).
        Low entropy  → energy concentrated at defect frequency harmonics (fault).

        Note: counterintuitive but correct. Healthy Gaussian noise scores HIGH
        entropy because energy is uniformly spread. A faulty bearing scores LOWER
        entropy because impulses create peaks at BPFO/BPFI harmonics.
        Use in combination with BPFO energy, not in isolation.

        Based on: Shannon entropy of normalized PSD.
        """
        psd_norm = psd / (np.sum(psd) + 1e-12)
        psd_norm = psd_norm[psd_norm > 0]
        se = -np.sum(psd_norm * np.log2(psd_norm))
        se_max = np.log2(len(psd))  # normalize
        return float(se / se_max if se_max > 0 else 0.0)

    def defect_freq_energy(
        self, freqs: np.ndarray, psd: np.ndarray, freq_hz: float, bandwidth_hz: float = 5.0
    ) -> float:
        """
        Compute energy around a defect frequency and its harmonics.

        This is the key physics insight: fault energy concentrates at BPFO/BPFI/BSF
        rather than spreading, because a localised defect is struck at a rate fixed
        by geometry and shaft speed.
        """
        total_energy = 0.0
        for harmonic in range(1, self.n_harmonics + 1):
            center = freq_hz * harmonic
            mask = (freqs >= center - bandwidth_hz) & (freqs <= center + bandwidth_hz)
            total_energy += np.sum(psd[mask])
        return float(total_energy)

    def high_freq_energy_ratio(
        self, freqs: np.ndarray, psd: np.ndarray, cutoff_hz: float = 5000.0
    ) -> float:
        """
        Compute the ratio of energy above a cutoff to total energy.

        Counterintuitive direction, and verified: healthy Gaussian noise is
        already high-frequency-rich, so this does not simply rise with damage.
        Read it against the defect-frequency band energy, not on its own.
        """
        total = np.sum(psd)
        high = np.sum(psd[freqs >= cutoff_hz])
        return float(high / total if total > 1e-12 else 0.0)

    def extract_frequency_domain(self, x: np.ndarray) -> dict:
        """Compute all frequency-domain features for one channel waveform."""
        freqs, psd = self.welch_psd(x)

        features = {
            "spectral_entropy": self.spectral_entropy(psd),
            "high_freq_ratio": self.high_freq_energy_ratio(freqs, psd),
            "bpfo_energy": self.defect_freq_energy(freqs, psd, self.defect_freqs["BPFO_Hz"]),
            "bpfi_energy": self.defect_freq_energy(freqs, psd, self.defect_freqs["BPFI_Hz"]),
            "bsf_energy": self.defect_freq_energy(freqs, psd, self.defect_freqs["BSF_Hz"]),
            "ftf_energy": self.defect_freq_energy(freqs, psd, self.defect_freqs["FTF_Hz"]),
            "dominant_freq_hz": float(freqs[np.argmax(psd)]),
            "psd_mean": float(np.mean(psd)),
            "psd_std": float(np.std(psd)),
        }

        return features

    def extract_from_raw(self, x: np.ndarray, channel_name: str = "ch") -> dict:
        """
        Extract every feature from a raw 1-D waveform.

        Combines the time-domain and frequency-domain sets, prefixing each key
        with ``channel_name`` so channels stay distinguishable once merged.
        """
        td = {f"{channel_name}_{k}": v for k, v in self.extract_time_domain(x).items()}
        fd = {f"{channel_name}_{k}": v for k, v in self.extract_frequency_domain(x).items()}
        return {**td, **fd}


# ─── Aggregate Feature Extraction from Processed CSVs ─────────────────────
# When raw files are large, we can also re-derive advanced features
# from the already-loaded statistical summaries.


def add_rolling_features(
    df: pd.DataFrame, columns: list, windows: list | None = None
) -> pd.DataFrame:
    """
    Add rolling statistics over time to capture degradation trend.

    Rolling mean / std / max show the *rate of change* of vibration,
    not just instantaneous value — critical for early detection.
    """
    # A literal default would be one list built at import and shared by every
    # call, so a caller that mutated it would change the default for the rest
    # of the process.
    if windows is None:
        windows = [10, 50]

    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(window=w, min_periods=1).std().fillna(0)
            df[f"{col}_roll_max_{w}"] = df[col].rolling(window=w, min_periods=1).max()

    return df


def add_change_rate_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Add first-order difference (rate of change) features.

    Sudden jumps in RMS or kurtosis are early warning signals, and a difference
    column exposes them to a detector that otherwise only sees levels.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_diff"] = df[col].diff().fillna(0)
            df[f"{col}_diff_abs"] = df[f"{col}_diff"].abs()
    return df


def enrich_processed(test_id: int) -> pd.DataFrame:
    """
    Load a processed CSV and add rolling and change-rate features.

    Writes the enriched table to ``data/processed/test{N}_features.csv``. That
    file is gitignored, so it is absent from a fresh clone until this runs --
    which is why ``detection.resolve_feature_table`` names the source it used
    instead of silently taking whichever table happens to exist.

    Parameters
    ----------
    test_id : int
        1, 2 or 3.

    Returns
    -------
    pandas.DataFrame
        The enriched table.
    """
    df = load_processed(test_id)

    # Select RMS columns (most informative for rolling)
    rms_cols = [c for c in df.columns if "_rms" in c]
    kurt_cols = [c for c in df.columns if "_kurt" in c]

    print(f"  Adding rolling features over windows [10, 50] for {len(rms_cols)} RMS channels...")
    df = add_rolling_features(df, rms_cols + kurt_cols, windows=[10, 50])

    print("  Adding rate-of-change features...")
    df = add_change_rate_features(df, rms_cols + kurt_cols)

    # Save
    out_path = TEST_CONFIG[test_id]["output_file"].parent / f"test{test_id}_features.csv"
    df.to_csv(out_path)
    print(f"Saved enriched features: {out_path}  ({len(df)} rows, {len(df.columns)} columns)")
    return df


# ─── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from NASA Bearing Dataset")
    parser.add_argument("--test", type=str, default="all")
    args = parser.parse_args()

    tests = [1, 2, 3] if args.test == "all" else [int(args.test)]
    for t in tests:
        print(f"\nFeature engineering for Test {t}...")
        enrich_processed(t)
