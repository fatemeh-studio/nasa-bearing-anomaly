"""
test_detection.py
Tests for the Isolation Forest anomaly detection pipeline.
Run: pytest tests/test_detection.py -v
"""

import numpy as np
import pandas as pd
import pytest

from nasa_bearing_anomaly.detection import IsolationForestDetector, select_features

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def normal_df():
    """
    Synthetic DataFrame mimicking a healthy bearing time series.
    100 samples with low-amplitude Gaussian features — tight cluster.
    """
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "Bearing1_ch1_rms": np.random.normal(0.1, 0.005, n),
            "Bearing1_ch1_kurt": np.random.normal(3.0, 0.1, n),
            "Bearing1_ch1_std": np.random.normal(0.08, 0.003, n),
            "Bearing1_ch1_max": np.random.normal(0.3, 0.01, n),
        }
    )


@pytest.fixture
def degraded_df(normal_df):
    """
    Normal data with 10 anomalous samples appended at the end —
    simulates a bearing that degrades near the end of the test.
    """
    np.random.seed(7)
    n_anom = 10
    anomalies = pd.DataFrame(
        {
            "Bearing1_ch1_rms": np.random.normal(1.5, 0.1, n_anom),  # 15× normal
            "Bearing1_ch1_kurt": np.random.normal(12.0, 0.5, n_anom),  # high kurtosis
            "Bearing1_ch1_std": np.random.normal(0.9, 0.05, n_anom),
            "Bearing1_ch1_max": np.random.normal(4.0, 0.2, n_anom),
        }
    )
    return pd.concat([normal_df, anomalies], ignore_index=True)


FEATURE_COLS = [
    "Bearing1_ch1_rms",
    "Bearing1_ch1_kurt",
    "Bearing1_ch1_std",
    "Bearing1_ch1_max",
]


# ── IsolationForestDetector ────────────────────────────────────────────────


class TestIsolationForestDetector:
    def test_fit_returns_self(self, normal_df):
        detector = IsolationForestDetector(contamination=0.05)
        result = detector.fit(normal_df, FEATURE_COLS)
        assert result is detector

    def test_predict_adds_required_columns(self, degraded_df):
        detector = IsolationForestDetector(contamination=0.05)
        result = detector.fit_predict(degraded_df, FEATURE_COLS)
        assert "anomaly_score" in result.columns
        assert "anomaly_label" in result.columns
        assert "is_anomaly" in result.columns

    def test_anomaly_label_values(self, degraded_df):
        """anomaly_label must only contain 1 (normal) or -1 (anomaly)."""
        detector = IsolationForestDetector(contamination=0.05)
        result = detector.fit_predict(degraded_df, FEATURE_COLS)
        assert set(result["anomaly_label"].unique()).issubset({1, -1})

    def test_is_anomaly_is_boolean(self, degraded_df):
        detector = IsolationForestDetector(contamination=0.05)
        result = detector.fit_predict(degraded_df, FEATURE_COLS)
        assert result["is_anomaly"].dtype == bool

    def test_detects_obvious_anomalies(self, degraded_df):
        """
        The last 10 rows are clearly anomalous (15× RMS, high kurtosis).
        At least 5 of them should be flagged.
        """
        detector = IsolationForestDetector(contamination=0.1)
        result = detector.fit_predict(degraded_df, FEATURE_COLS)
        n_flagged_in_tail = result.iloc[-10:]["is_anomaly"].sum()
        assert n_flagged_in_tail >= 5, f"Expected ≥5 anomalies in tail, got {n_flagged_in_tail}"

    def test_normal_data_mostly_clean(self, normal_df):
        """
        On purely normal data with contamination=0.05,
        no more than 10% of samples should be flagged.
        """
        detector = IsolationForestDetector(contamination=0.05)
        result = detector.fit_predict(normal_df, FEATURE_COLS)
        fraction_flagged = result["is_anomaly"].mean()
        assert fraction_flagged <= 0.10, (
            f"Too many false positives on clean data: {fraction_flagged:.1%}"
        )

    def test_anomaly_score_is_finite(self, degraded_df):
        detector = IsolationForestDetector(contamination=0.05)
        result = detector.fit_predict(degraded_df, FEATURE_COLS)
        assert result["anomaly_score"].isna().sum() == 0
        assert np.isfinite(result["anomaly_score"].values).all()

    def test_pca_off_still_works(self, degraded_df):
        """use_pca=False should work without error."""
        detector = IsolationForestDetector(contamination=0.05, use_pca=False)
        result = detector.fit_predict(degraded_df, FEATURE_COLS)
        assert "is_anomaly" in result.columns

    def test_original_df_not_mutated(self, degraded_df):
        """fit_predict must not modify the input DataFrame."""
        original_cols = list(degraded_df.columns)
        detector = IsolationForestDetector(contamination=0.05)
        _ = detector.fit_predict(degraded_df, FEATURE_COLS)
        assert list(degraded_df.columns) == original_cols

    def test_save_and_load(self, degraded_df, tmp_path):
        """Saved model should produce identical predictions after loading."""
        detector = IsolationForestDetector(contamination=0.05)
        result_before = detector.fit_predict(degraded_df, FEATURE_COLS)

        detector.save(tmp_path)
        loaded = IsolationForestDetector.load(tmp_path)
        result_after = loaded.predict(degraded_df)

        pd.testing.assert_series_equal(
            result_before["is_anomaly"].reset_index(drop=True),
            result_after["is_anomaly"].reset_index(drop=True),
            check_names=False,
        )


# ── select_features ────────────────────────────────────────────────────────


class TestSelectFeatures:
    def test_returns_list(self, degraded_df):
        cols = select_features(degraded_df)
        assert isinstance(cols, list)

    def test_no_result_columns_included(self, degraded_df):
        """anomaly_score / is_anomaly should never appear in feature list."""
        df = degraded_df.copy()
        df["anomaly_score"] = 0.0
        df["is_anomaly"] = False
        df["anomaly_label"] = 1
        cols = select_features(df)
        for forbidden in ["anomaly_score", "is_anomaly", "anomaly_label"]:
            assert forbidden not in cols, f"'{forbidden}' should not be a feature"

    def test_bearing_prefix_filters_correctly(self, degraded_df):
        cols = select_features(degraded_df, bearing_prefix="Bearing1")
        assert all("Bearing1" in c for c in cols), "All selected features should belong to Bearing1"

    def test_nonexistent_prefix_falls_back(self, degraded_df):
        """
        If the named prefix matches nothing, select_features falls back
        to all available indicator columns regardless of bearing prefix.
        """
        df = degraded_df.copy()
        df["global_ch1_rms"] = 0.1  # matches _rms keyword, no bearing prefix
        cols = select_features(df, bearing_prefix="Bearing99")
        assert "global_ch1_rms" in cols
