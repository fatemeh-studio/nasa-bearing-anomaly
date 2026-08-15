"""
Feature extraction for bearing health monitoring.

Two independent halves, and only the second is wired into the detection pipeline.

``FeatureExtractor`` operates on a **raw 1-D waveform**:

1. Time domain — RMS, excess kurtosis, skewness, peak, energy, and the crest, shape,
   impulse and clearance factors.
2. Frequency domain, from a Welch power spectral density (PSD) estimate — band energy at
   BPFO, BPFI and BSF harmonics, spectral entropy, and the ratio of energy above 5 kHz.
   FTF gets no band: at the working resolution its band would reach DC. See
   ``FeatureExtractor.band_is_resolvable``.
3. Envelope domain — the same defect bands measured on the spectrum of the amplitude
   envelope of the 2-10 kHz resonance region, which is where a bearing defect actually
   makes itself visible. A defect modulates a resonance rather than radiating at its own
   frequency, so the line is in the envelope, not the raw spectrum.

Spectral entropy is **higher** in a healthy bearing, not lower: healthy noise is broadband
and near-flat, which is maximum entropy, and a defect concentrates energy into BPFO
harmonics, which lowers it. Pinned by ``test_spectral_entropy_faulty_lower_than_healthy``.

``enrich_processed`` operates on the **summary table** ``loading.py`` writes, adding
rolling and first-difference columns over the RMS and kurtosis channels.

**Only ``enrich_processed`` reaches the model.** The detection pipeline reads the summary
table and never a waveform, so nothing ``FeatureExtractor`` computes appears in the feature
matrix. It is exercised by the test suite and demonstrated on synthetic signals in
notebook 02. Joining the two costs one pass over the raw archive.

Usage:
    from nasa_bearing_anomaly.features import FeatureExtractor
    extractor = FeatureExtractor()
    features = extractor.extract_from_raw(waveform_array)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .config import BEARING_PARAMS, SAMPLING_RATE_HZ, TEST_CONFIG, repo_path
from .loading import acquisition_files, load_processed, load_single_file, tqdm
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
    envelope_highpass_hz : float
        Lower edge in Hz of the resonance band used for envelope analysis. The
        upper edge is Nyquist, so at the dataset's 20 kHz this is a 2-10 kHz band.
    """

    def __init__(
        self,
        fs: int = SAMPLING_RATE_HZ,
        window_sec: float = 0.1,
        bearing_params: dict = BEARING_PARAMS,
        envelope_highpass_hz: float = 2000.0,
    ):
        self.fs = fs
        self.nperseg = int(window_sec * fs)
        self.defect_freqs = compute_defect_frequencies(bearing_params)

        # Harmonic orders to check for each defect frequency
        self.n_harmonics = 3

        # Welch bin spacing. Every band below is sized against it, because a band
        # narrower than one bin is a point sample rather than an integral.
        # Measured 2026-08-15: the previous fixed 5 Hz half-width resolved to
        # exactly one bin at every harmonic of every defect frequency, which made
        # the parameter inert below 10 Hz and left ftf_energy with a
        # healthy-to-faulty separation of 0.02 on real Test 3 data.
        self.freq_resolution_hz = fs / self.nperseg

        # Three bins is the narrowest span that is still an integral, and it
        # absorbs the shaft-speed drift a rig shows between acquisitions: the
        # defect frequencies are proportional to shaft speed, so a fixed band has
        # to be wide enough that the line does not walk out of it.
        self.band_halfwidth_hz = 1.5 * self.freq_resolution_hz

        # A band that would reach DC is dropped rather than clipped. Clipping
        # would make it silently asymmetric and fold the DC bin -- which carries
        # the signal mean -- into a feature meant to measure a defect line.
        self.band_energy_freqs = tuple(
            name
            for name in ("BPFO_Hz", "BPFI_Hz", "BSF_Hz", "FTF_Hz")
            if self.band_is_resolvable(self.defect_freqs[name])
        )

        # Fixed in advance rather than tuned: 2 kHz to Nyquist sits above every
        # defect fundamental and its first ten harmonics, so the passband holds
        # resonance-borne energy and none of the defect lines themselves.
        # Choosing it from the faulty files instead would be selection on the
        # outcome. A kurtogram would pick it per signal; that is the next step.
        self.envelope_highpass_hz = envelope_highpass_hz

    def band_is_resolvable(self, freq_hz: float) -> bool:
        """
        Whether a defect band centred on ``freq_hz`` clears DC.

        The band is ``freq_hz`` plus and minus ``band_halfwidth_hz``. A centre
        below the half-width puts 0 Hz inside the band, and the DC bin carries the
        signal mean rather than defect energy.

        Parameters
        ----------
        freq_hz : float
            Band centre in Hz.

        Returns
        -------
        bool
            True when the band lies entirely above 0 Hz.
        """
        return freq_hz - self.band_halfwidth_hz > 0

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
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        freq_hz: float,
        bandwidth_hz: float | None = None,
    ) -> float:
        """
        Compute energy around a defect frequency and its harmonics.

        This is the key physics insight: fault energy concentrates at BPFO/BPFI/BSF
        rather than spreading, because a localised defect is struck at a rate fixed
        by geometry and shaft speed.

        Parameters
        ----------
        freqs, psd : numpy.ndarray
            Frequency grid in Hz and the power spectral density on it, from
            ``welch_psd`` or ``envelope_spectrum``.
        freq_hz : float
            Defect frequency in Hz. Harmonics 1 to ``n_harmonics`` are summed.
        bandwidth_hz : float, optional
            Half-width of each band. Defaults to ``band_halfwidth_hz``, three PSD
            bins wide. A value below ``freq_resolution_hz`` collapses the band to
            a single bin and makes this a point sample rather than an integral.

        Returns
        -------
        float
            Summed power over the harmonic bands, in the PSD's units.
        """
        if bandwidth_hz is None:
            bandwidth_hz = self.band_halfwidth_hz
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
        """
        Compute all frequency-domain features for one channel waveform.

        A band-energy column appears only for a defect frequency whose band clears
        DC at the working resolution, so FTF is absent at the default window. See
        ``band_is_resolvable``.
        """
        freqs, psd = self.welch_psd(x)

        features = {
            "spectral_entropy": self.spectral_entropy(psd),
            "high_freq_ratio": self.high_freq_energy_ratio(freqs, psd),
            "dominant_freq_hz": float(freqs[np.argmax(psd)]),
            "psd_mean": float(np.mean(psd)),
            "psd_std": float(np.std(psd)),
        }
        for name in self.band_energy_freqs:
            key = f"{name.replace('_Hz', '').lower()}_energy"
            features[key] = self.defect_freq_energy(freqs, psd, self.defect_freqs[name])

        return features

    # ── Envelope (High-Frequency Resonance) Features ───────────────────────

    def envelope(self, x: np.ndarray) -> np.ndarray:
        """
        Amplitude envelope of the waveform's high-frequency content.

        A localised defect does not radiate at the defect frequency. Each impact
        rings a structural resonance in the kHz range, so the defect rate appears
        as the *modulation* of that resonance and carries no line of its own in the
        raw spectrum. Band-passing to the resonance region and taking the
        analytic-signal magnitude moves that modulation to baseband, where it is a
        line again. Standard practice for rolling-element bearings; see Randall and
        Antoni (2011), *Mechanical Systems and Signal Processing* 25(2).

        Parameters
        ----------
        x : numpy.ndarray
            Raw 1-D waveform.

        Returns
        -------
        numpy.ndarray
            Non-negative envelope, same length as ``x``. The first and last few
            hundred samples ring from the zero-phase filter and should not be read
            on their own.
        """
        sos = signal.butter(
            4, self.envelope_highpass_hz, btype="highpass", fs=self.fs, output="sos"
        )
        return np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))

    def envelope_spectrum(self, env: np.ndarray):
        """
        Welch power spectral density of an envelope, with its mean removed.

        The envelope is strictly positive and so carries a large DC component. Left
        in, it would dominate every band and swamp the defect line the spectrum
        exists to expose.

        Parameters
        ----------
        env : numpy.ndarray
            Envelope, from ``envelope``.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            Frequencies in Hz, and the envelope power spectral density.
        """
        return signal.welch(
            env - np.mean(env),
            fs=self.fs,
            nperseg=self.nperseg,
            window="hann",
            average="mean",
        )

    def extract_envelope_domain(self, x: np.ndarray) -> dict:
        """
        Defect-band energy measured on the envelope spectrum rather than the raw one.

        Uses the same bands and the same resolvability rule as
        ``extract_frequency_domain``, so a frequency excluded there is excluded
        here. ``env_kurtosis`` is included separately because a periodic impact
        train makes the envelope itself impulsive, wherever its energy sits.
        """
        env = self.envelope(x)
        freqs, psd = self.envelope_spectrum(env)

        features = {"env_kurtosis": self.kurtosis(env)}
        for name in self.band_energy_freqs:
            key = f"env_{name.replace('_Hz', '').lower()}_energy"
            features[key] = self.defect_freq_energy(freqs, psd, self.defect_freqs[name])

        return features

    def extract_from_raw(self, x: np.ndarray, channel_name: str = "ch") -> dict:
        """
        Extract every feature from a raw 1-D waveform.

        Combines the time-domain, frequency-domain and envelope sets, prefixing
        each key with ``channel_name`` so channels stay distinguishable once
        merged into one row per acquisition.
        """
        td = {f"{channel_name}_{k}": v for k, v in self.extract_time_domain(x).items()}
        fd = {f"{channel_name}_{k}": v for k, v in self.extract_frequency_domain(x).items()}
        ed = {f"{channel_name}_{k}": v for k, v in self.extract_envelope_domain(x).items()}
        return {**td, **fd, **ed}


# ─── Waveform Pass over the Raw Archive ────────────────────────────────────

# Keys `extract_from_raw` shares with the summary table loading.py writes --
# either under the same name (rms, std) or a different name for the same
# quantity (kurtosis/_kurt, peak/_max, skewness/_skew). Dropped from the
# spectral table so a join cannot yield two columns for one measurement, and so
# that `_kurtosis` cannot be picked up a second time by select_features'
# `_kurt` substring filter, which would silently reweight the time domain.
SUMMARY_DUPLICATE_KEYS = ("rms", "std", "kurtosis", "peak", "skewness")


def extract_spectral(test_id: int, max_files: int = None, verbose: bool = True) -> pd.DataFrame:
    """
    Compute waveform features for every acquisition in one test run.

    This is the join the summary pipeline lacks. ``loading.load_test`` reduces
    each acquisition to six statistics and discards the waveform, so no
    frequency- or envelope-domain feature can reach the model from that table.
    This walks the raw archive a second time and keeps what ``FeatureExtractor``
    computes, one row per acquisition indexed by ``file_index`` so it joins the
    summary table row for row.

    Costs one pass over the 6.2 GB archive, measured at roughly five minutes for
    all three tests. Columns already carried by the summary table are dropped --
    see ``SUMMARY_DUPLICATE_KEYS``.

    Parameters
    ----------
    test_id : int
        1, 2 or 3.
    max_files : int, optional
        Limit the number of files. As in ``loading.py`` this also suppresses the
        write, so a smoke run cannot replace a full table with a partial one.
    verbose : bool
        Show a progress bar and the band configuration in use.

    Returns
    -------
    pandas.DataFrame
        One row per acquisition, indexed by ``file_index``.
    """
    files, columns = acquisition_files(test_id, max_files)
    extractor = FeatureExtractor()

    if verbose:
        bands = ", ".join(n.replace("_Hz", "") for n in extractor.band_energy_freqs)
        print(f"\nExtracting waveform features for Test {test_id}: {len(files)} files")
        print(f"   Bands: {bands}, half-width {extractor.band_halfwidth_hz:.1f} Hz")
        print(f"   Envelope band: {extractor.envelope_highpass_hz:.0f} Hz to Nyquist")

    records = []
    for i, filepath in enumerate(tqdm(files, desc=f"Test {test_id}", disable=not verbose)):
        try:
            raw = load_single_file(filepath)
            record = {"file_index": i}

            # The channel guard ran in acquisition_files, before this loop, so a
            # mismatch cannot be swallowed by the except below.
            for j, col in enumerate(columns):
                prefix = f"{col}_"
                for key, value in extractor.extract_from_raw(raw[:, j], channel_name=col).items():
                    if key[len(prefix) :] not in SUMMARY_DUPLICATE_KEYS:
                        record[key] = value

            records.append(record)

        except Exception as e:
            if verbose:
                print(f"  Warning: Skipped {filepath.name}: {e}")
            continue

    if len(records) != len(files):
        print(f"Warning: {len(files) - len(records)} of {len(files)} files were skipped.")

    df = pd.DataFrame(records).set_index("file_index")

    if max_files:
        print("Skipped save: --max_files was set, so this table is partial.")
        return df

    out_path = TEST_CONFIG[test_id]["output_file"].parent / f"test{test_id}_spectral.csv"
    # Six significant figures: these are derived quantities whose inputs carry
    # far less precision than that, and full repr doubles the committed file.
    df.to_csv(out_path, float_format="%.6g")
    print(f"Saved: {repo_path(out_path)}  ({len(df)} rows, {len(df.columns)} columns)")
    return df


# ─── Rolling and Difference Features over the Summary Table ────────────────
# This half of the module is what the detection pipeline reads. It adds temporal
# context -- rate of change rather than level -- to the per-file statistics, and
# needs only the committed summary tables, not the raw archive a clone lacks.


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

    new = {}
    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            roll = df[col].rolling(window=w, min_periods=1)
            new[f"{col}_roll_mean_{w}"] = roll.mean()
            new[f"{col}_roll_std_{w}"] = roll.std().fillna(0)
            new[f"{col}_roll_max_{w}"] = roll.max()

    # Concatenated once rather than assigned one column at a time. An ablation arm
    # adds up to 64 columns here, and repeated insertion both fragments the frame
    # and makes pandas emit a PerformanceWarning -- which is captured into notebook
    # output and rendered on the site.
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_change_rate_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Add first-order difference (rate of change) features.

    Sudden jumps in RMS or kurtosis are early warning signals, and a difference
    column exposes them to a detector that otherwise only sees levels.
    """
    new = {}
    for col in columns:
        if col in df.columns:
            diff = df[col].diff().fillna(0)
            new[f"{col}_diff"] = diff
            new[f"{col}_diff_abs"] = diff.abs()
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


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
    print(
        f"Saved enriched features: {repo_path(out_path)}  ({len(df)} rows, {len(df.columns)} columns)"
    )
    return df


# ─── Ablation Arms ─────────────────────────────────────────────────────────

# Feature sets compared in the ablation. 'enriched' -- the published run -- is
# deliberately NOT built here. It keeps its original path through
# detection.resolve_feature_table, so the regression guard compares against the
# code that produced the published numbers rather than a reimplementation of it.
ARMS = ("log", "psd", "psd_enriched", "envelope")

# Strictly-positive scale quantities, which belong on a log scale: they are
# multiplicative and span orders of magnitude, and the field's own unit for them,
# the decibel, is already a logarithm. It matters to the model rather than only to
# the eye -- an isolation tree draws its split points uniformly across a feature's
# linear range, so a feature spanning three decades gets nearly all of its splits
# inside the top decade.
#
# Named rather than detected from the values. A rule like "log whatever happens to
# be positive" would treat one feature differently between tests and give the three
# runs different schemas. Signed quantities (_kurt, _mean, _skew) and bounded ratios
# (spectral_entropy, high_freq_ratio, the shape factors) are excluded by the rule,
# not by how they perform.
LOG_SUFFIXES = ("_rms", "_std", "_max", "_mean_abs", "_energy", "_psd_mean")

# The Welch-PSD block. Listed explicitly rather than matched loosely, because
# '_bpfo_energy' is also the ending of '_env_bpfo_energy'.
PSD_SUFFIXES = (
    "_spectral_entropy",
    "_high_freq_ratio",
    "_dominant_freq_hz",
    "_psd_mean",
    "_psd_std",
    "_bpfo_energy",
    "_bpfi_energy",
    "_bsf_energy",
)

TIME_SUFFIXES = ("_rms", "_kurt", "_std", "_max")
ROLL_SUFFIXES = ("_rms", "_kurt")


def spectral_path(test_id: int) -> Path:
    """Path to the committed waveform-feature table for one test."""
    return TEST_CONFIG[test_id]["output_file"].parent / f"test{test_id}_spectral.csv"


def add_log_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add a base-10 log companion for every strictly-positive scale column.

    A companion rather than a replacement, and this is load-bearing:
    ``business.find_shutdown_tail`` and ``business.check_healthy_region`` both
    compare RMS against a fraction of its own baseline, and a ratio of logarithms
    is not the logarithm of a ratio. Overwriting ``_rms`` in place would make a
    stopped rig read as twice baseline instead of a sixteenth of it, and the
    post-shutdown tail would silently stop being found.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature frame. Not modified in place.

    Returns
    -------
    tuple of (pandas.DataFrame, list of str)
        A copy carrying the new ``{column}_log10`` columns, and their names.
    """
    out = df.copy()
    added = []
    for col in df.columns:
        if col.endswith(LOG_SUFFIXES):
            name = f"{col}_log10"
            # Floored rather than a bare log10: a channel reading exactly zero
            # would give -inf, and the detector fills missing values with 0, which
            # on this scale reads as 1.0 -- orders of magnitude above any healthy
            # value. Measured across all three tests, nothing falls below 1e-12,
            # so the floor does not bind on this data.
            out[name] = np.log10(np.maximum(df[col].to_numpy(dtype=float), 1e-12))
            added.append(name)
    return out, added


def _scored_name(column: str) -> str:
    """Log companion of a column where it has one, otherwise the column itself."""
    return f"{column}_log10" if column.endswith(LOG_SUFFIXES) else column


def build_arm(test_id: int, arm: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Assemble one ablation arm: the feature frame, and the columns it scores.

    Scored columns are named explicitly rather than matched by keyword.
    ``detection.select_features`` matches ``_std`` and ``_kurt`` as substrings, so
    on a joined frame it would pull ``psd_std`` and ``env_kurtosis`` into arms that
    exist to exclude them -- and it degrades to a smaller set rather than raising,
    so nothing would go red.

    The arms:

    ``log``
        The published feature set, scored on a log scale. Comparing it against the
        published run isolates what the change of scale alone is worth, so a later
        gain cannot be credited to the physics when it came from the units.
    ``psd``
        ``log`` plus the Welch-PSD block: defect-band energy at BPFO, BPFI and BSF,
        spectral entropy, high-frequency ratio and the PSD moments.
    ``psd_enriched``
        ``psd`` with the same rolling and difference treatment the time-domain
        columns already get, so a null result cannot be blamed on the spectral
        features having been denied the temporal context the others have.
    ``envelope``
        ``log`` plus the same three defect bands measured on the envelope spectrum,
        plus the envelope's kurtosis. Parallel to ``psd`` by construction, so the
        two differ only in how the same defect frequencies are estimated.

    Parameters
    ----------
    test_id : int
        1, 2 or 3.
    arm : {'log', 'psd', 'psd_enriched', 'envelope'}
        Which feature set to assemble.

    Returns
    -------
    tuple of (pandas.DataFrame, list of str)
        The joined frame, which keeps ``timestamp`` and the linear columns that
        ``business.py`` reads, and every column this arm scores across all
        bearings. The caller narrows to one bearing by prefix.

    Raises
    ------
    ValueError
        If ``arm`` is not one of ``ARMS``.
    FileNotFoundError
        If a spectral arm is requested and the waveform table is absent.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")

    summary = load_processed(test_id)
    meta = [c for c in ("timestamp", "filename") if c in summary.columns]
    base = [c for c in summary.columns if c.endswith(TIME_SUFFIXES)]

    frame, _ = add_log_columns(summary[meta + base])
    scored = [_scored_name(c) for c in base]

    # Rolling and difference columns over RMS and kurtosis only, matching
    # enrich_processed, so this block is the published one up to the change of
    # scale. Derived AFTER the log, so a difference of logs is a growth ratio --
    # the right rate of change for a multiplicative quantity.
    roll_src = [_scored_name(c) for c in base if c.endswith(ROLL_SUFFIXES)]
    before = set(frame.columns)
    frame = add_rolling_features(frame, roll_src, windows=[10, 50])
    frame = add_change_rate_features(frame, roll_src)
    scored += [c for c in frame.columns if c not in before]

    if arm == "log":
        return frame, scored

    path = spectral_path(test_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Waveform feature table not found: {path}\n"
            f"Run: python -m nasa_bearing_anomaly.features --test {test_id} --spectral"
        )
    spectral = pd.read_csv(path, index_col="file_index")

    if arm == "envelope":
        block_base = [c for c in spectral.columns if "_env_" in c]
    else:
        block_base = [c for c in spectral.columns if c.endswith(PSD_SUFFIXES) and "_env_" not in c]

    block, _ = add_log_columns(spectral[block_base])
    block_scored = [_scored_name(c) for c in block_base]

    if arm == "psd_enriched":
        before = set(block.columns)
        block = add_rolling_features(block, block_scored, windows=[10, 50])
        block = add_change_rate_features(block, block_scored)
        block_scored += [c for c in block.columns if c not in before]

    return frame.join(block), scored + block_scored


# ─── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from NASA Bearing Dataset")
    parser.add_argument("--test", type=str, default="all")
    parser.add_argument(
        "--spectral",
        action="store_true",
        help="Walk the raw archive and write test{N}_spectral.csv, instead of the "
        "rolling and difference pass over the committed summary tables",
    )
    parser.add_argument(
        "--max_files", type=int, default=None, help="Limit files; suppresses the save"
    )
    args = parser.parse_args()

    tests = [1, 2, 3] if args.test == "all" else [int(args.test)]
    for t in tests:
        if args.spectral:
            extract_spectral(t, max_files=args.max_files)
        else:
            print(f"\nFeature engineering for Test {t}...")
            enrich_processed(t)
