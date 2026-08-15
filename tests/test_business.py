"""
Tests for the business-outcome layer.

The fixtures here are synthetic on purpose. A test that reads the real tables
would pin whatever the pipeline currently produces, so a regression in detection
would quietly rewrite the expected values along with the actual ones. A hand-built
run whose failure point is known by construction fails when the metric changes.
"""

import numpy as np
import pandas as pd
import pytest

from nasa_bearing_anomaly.business import (
    HEALTHY_RMS_RATIO,
    KM_GRID,
    DowntimeCostModel,
    LeadTimeResult,
    calibrate_threshold,
    check_healthy_region,
    compute_lead_time,
    cost_sensitivity,
    drop_shutdown_tail,
    find_shutdown_tail,
    flag_files,
    healthy_windows,
    rms_columns,
    select_km,
    sustained_alarm,
    writes_published_summary,
)
from nasa_bearing_anomaly.features import ARMS

BEARINGS = ["Bearing1", "Bearing2", "Bearing3", "Bearing4"]


def make_run(
    n_live: int = 1000,
    n_shutdown: int = 1,
    degrade_from: int = 800,
    baseline_rms: float = 0.07,
    peak_rms: float = 0.6,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a synthetic run: healthy, then degrading, then optionally post-shutdown.

    Scores follow the IsolationForest convention -- higher is healthier -- so
    degradation drives ``anomaly_score`` downward.
    """
    rng = np.random.default_rng(seed)
    n = n_live + n_shutdown

    ramp = np.zeros(n_live)
    if degrade_from < n_live:
        span = n_live - degrade_from
        ramp[degrade_from:] = np.linspace(0, 1, span)

    rms = baseline_rms + rng.normal(0, baseline_rms * 0.05, n_live) + ramp * peak_rms
    score = rng.normal(0.15, 0.03, n_live) - ramp * 0.45

    if n_shutdown:
        rms = np.concatenate([rms, np.full(n_shutdown, 0.004)])
        score = np.concatenate([score, rng.normal(0.15, 0.03, n_shutdown)])

    stamps = pd.date_range("2004-03-04 09:27:46", periods=n, freq="10min")
    data = {"timestamp": stamps, "anomaly_score": score}
    for b in BEARINGS:
        data[f"{b}_ch1_rms"] = rms
    df = pd.DataFrame(data)
    df.index = pd.RangeIndex(n, name="file_index")
    return df


# ─── Post-Shutdown Tail ────────────────────────────────────────────────────


class TestShutdownTail:
    """The tail is measured, never assumed to be one file."""

    @pytest.mark.parametrize("n_shutdown", [0, 1, 2, 5])
    def test_tail_length_is_measured(self, n_shutdown):
        df = make_run(n_shutdown=n_shutdown)
        assert find_shutdown_tail(df) == n_shutdown

    def test_run_ending_live_keeps_every_file(self):
        """Test 1 ends on its RMS peak; dropping a file would discard the failure."""
        df = make_run(n_shutdown=0)
        live, dropped = drop_shutdown_tail(df)
        assert dropped == 0
        assert len(live) == len(df)
        assert live.index[-1] == df.index[-1]

    def test_two_file_tail_is_fully_removed(self):
        """Test 2's tail is two files; dropping only the last leaves a dead rig."""
        df = make_run(n_shutdown=2)
        live, dropped = drop_shutdown_tail(df)
        assert dropped == 2
        assert live[rms_columns(live)].max(axis=1).iloc[-1] > 0.1

    def test_missing_rms_columns_raises(self):
        df = make_run().drop(columns=[f"{b}_ch1_rms" for b in BEARINGS])
        with pytest.raises(ValueError, match="_ch1_rms"):
            find_shutdown_tail(df)

    def test_single_quiet_bearing_is_not_shutdown(self):
        """One bearing going quiet is damage; shutdown means all channels stop."""
        df = make_run(n_shutdown=0)
        df.loc[df.index[-3:], "Bearing2_ch1_rms"] = 0.001
        assert find_shutdown_tail(df) == 0


# ─── Sustained-Alert Rule ──────────────────────────────────────────────────


class TestSustainedAlarm:
    """k-of-last-m semantics, including the unfillable leading window."""

    def test_first_m_minus_one_positions_never_trigger(self):
        flagged = np.ones(20, dtype=bool)
        out = sustained_alarm(flagged, k=5, m=5)
        assert not out[:4].any()
        assert out[4:].all()

    def test_requires_k_within_the_window(self):
        flagged = np.array([True, True, False, False, False, False, False, False])
        assert not sustained_alarm(flagged, k=3, m=5).any()

    def test_tolerates_gaps(self):
        """8-of-10 must survive two clean windows; a fault need not flag every file."""
        flagged = np.array([True] * 4 + [False] + [True] * 4 + [False] + [True] * 4)
        assert sustained_alarm(flagged, k=8, m=10).any()

    def test_all_clean_never_triggers(self):
        assert not sustained_alarm(np.zeros(100, dtype=bool), k=3, m=5).any()


class TestThresholdCalibration:
    """The threshold is what sets the alarm rate, replacing contamination."""

    def test_target_far_is_achieved_on_calibration_data(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(0.15, 0.03, 20_000)
        thr = calibrate_threshold(scores, target_far=0.01)
        assert 0.008 < float((scores < thr).mean()) < 0.012

    def test_lower_target_gives_stricter_threshold(self):
        rng = np.random.default_rng(2)
        scores = rng.normal(0.15, 0.03, 20_000)
        assert calibrate_threshold(scores, 0.001) < calibrate_threshold(scores, 0.05)

    def test_reconstruction_error_scores_calibrate_on_the_upper_tail(self):
        """
        The autoencoder scores error, so high is anomalous -- the opposite of the
        Isolation Forest. Calibrating on the wrong tail would flag the healthiest
        files while reporting a plausible-looking rate.
        """
        rng = np.random.default_rng(3)
        scores = rng.normal(0.15, 0.03, 20_000)
        thr = calibrate_threshold(scores, 0.01, higher_is_anomalous=True)
        assert 0.008 < float((scores > thr).mean()) < 0.012
        assert thr > calibrate_threshold(scores, 0.01, higher_is_anomalous=False)

    def test_flag_files_comparison_follows_the_convention(self):
        scores = np.array([-1.0, 0.0, 1.0])
        assert flag_files(scores, 0.0).tolist() == [True, False, False]
        assert flag_files(scores, 0.0, higher_is_anomalous=True).tolist() == [
            False,
            False,
            True,
        ]

    def test_inverted_scores_give_the_same_lead_time(self):
        """Negating the scores and flipping the flag must be a no-op end to end."""
        df = make_run()
        base = compute_lead_time(df, test_id=3)
        flipped = df.copy()
        flipped["anomaly_score"] = -flipped["anomaly_score"]
        other = compute_lead_time(flipped, test_id=3, higher_is_anomalous=True)
        assert other.alarm_file == base.alarm_file
        assert other.lead_hours == pytest.approx(base.lead_hours)


class TestSelectKm:
    """k and m are chosen against healthy data only, never against lead time."""

    def test_picks_first_clean_rule_in_grid_order(self):
        flagged = np.zeros(1000, dtype=bool)
        windows = healthy_windows(1000)
        assert select_km(flagged, windows) == KM_GRID[0]

    def test_skips_rules_that_fire_on_healthy_data(self):
        flagged = np.zeros(1000, dtype=bool)
        windows = healthy_windows(1000)
        # A solid block of flags inside the selection window defeats short rules.
        flagged[windows.selection.start : windows.selection.start + 12] = True
        k, m = select_km(flagged, windows)
        assert (k, m) != KM_GRID[0]
        assert not sustained_alarm(flagged, k, m)[windows.selection].any()

    def test_a_burst_in_the_calibration_window_also_defeats_short_rules(self):
        """
        Regression: test 3 opens with an anomalous burst inside the calibration
        window. Judging the rule on the selection window alone left that burst
        untested, so 3-of-5 was accepted and the first sustained alarm landed at
        file 28 -- a 1068 h "lead time" that is just the length of the run.
        """
        flagged = np.zeros(1000, dtype=bool)
        windows = healthy_windows(1000)
        assert windows.calibration.stop > 12  # the burst really is pre-selection
        flagged[5:17] = True
        k, m = select_km(flagged, windows)
        assert (k, m) != KM_GRID[0]
        region = slice(0, windows.selection.stop)
        assert not sustained_alarm(flagged, k, m)[region].any()

    def test_returns_none_when_no_rule_is_clean(self):
        flagged = np.ones(1000, dtype=bool)
        assert select_km(flagged, healthy_windows(1000)) is None

    def test_selection_ignores_the_held_out_window(self):
        """Alarms after the selection window must not influence the choice."""
        flagged = np.zeros(1000, dtype=bool)
        windows = healthy_windows(1000)
        flagged[windows.held_out] = True
        assert select_km(flagged, windows) == KM_GRID[0]


class TestHealthyWindows:
    """The three windows must stay disjoint or the reported rate is circular."""

    def test_windows_are_ordered_and_disjoint(self):
        w = healthy_windows(1000)
        assert w.calibration.stop == w.selection.start
        assert w.selection.stop == w.held_out.start
        assert w.calibration.start == 0
        assert w.held_out.stop <= 1000

    def test_short_series_raises_rather_than_returning_empty_windows(self):
        with pytest.raises(ValueError, match="too short"):
            healthy_windows(5)


class TestCheckHealthyRegion:
    """The healthy assumption is verified, not trusted."""

    def test_late_degradation_passes(self):
        df = make_run(degrade_from=800, n_shutdown=0)
        ok, ratio = check_healthy_region(df, "Bearing3")
        assert ok
        assert ratio <= HEALTHY_RMS_RATIO

    def test_early_degradation_is_flagged(self):
        df = make_run(degrade_from=100, n_shutdown=0)
        ok, ratio = check_healthy_region(df, "Bearing3")
        assert not ok
        assert ratio > HEALTHY_RMS_RATIO


# ─── End To End ────────────────────────────────────────────────────────────


class TestComputeLeadTime:
    """The whole chain, on a run whose failure point is known by construction."""

    def test_lead_time_matches_the_built_in_failure_point(self):
        df = make_run(n_live=1000, n_shutdown=1, degrade_from=800)
        r = compute_lead_time(df, test_id=3, feature_source="synthetic")

        assert isinstance(r, LeadTimeResult)
        assert r.n_shutdown_dropped == 1
        assert r.n_live == 1000
        # The alarm must land in the degrading region, not before it.
        assert r.alarm_file is not None
        assert 795 <= r.alarm_file <= 900
        assert not r.alarm_inside_healthy
        # Anchor is the last live file, not the post-shutdown one.
        assert r.anchor_file == 999
        # 10-minute acquisitions: lead time follows from the file gap.
        expected = (999 - r.alarm_file) * 10 / 60
        assert r.lead_hours == pytest.approx(expected, abs=0.02)

    def test_false_alarm_rate_is_reported_out_of_sample(self):
        r = compute_lead_time(make_run(), test_id=3)
        assert r.heldout_files > 0
        assert r.heldout_hours > 0
        assert r.heldout_alarms == 0

    def test_zero_alarms_is_reported_as_an_upper_bound(self):
        """A bare '0.0%' reads as a result without being one."""
        r = compute_lead_time(make_run(), test_id=3)
        assert "upper bound" in r.far_statement
        assert str(r.heldout_files) in r.far_statement
        assert "0.00%" not in r.far_statement

    def test_healthy_run_yields_no_lead_time_rather_than_a_flattering_one(self):
        df = make_run(degrade_from=10_000)  # never degrades
        r = compute_lead_time(df, test_id=3)
        assert r.lead_hours is None
        assert r.alarm_file is None
        assert any("never triggers" in n for n in r.notes)

    def test_run_ending_live_anchors_on_the_last_file(self):
        df = make_run(n_shutdown=0)
        r = compute_lead_time(df, test_id=3)
        assert r.n_shutdown_dropped == 0
        assert r.anchor_file == 999
        assert any("No post-shutdown tail" in n for n in r.notes)

    def test_missing_columns_raise(self):
        df = make_run().drop(columns=["anomaly_score"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_lead_time(df, test_id=3)

    def test_unparseable_timestamps_raise_rather_than_silently_dropping(self):
        df = make_run()
        df["timestamp"] = df["timestamp"].astype(object)
        df.iloc[5, df.columns.get_loc("timestamp")] = "not-a-date"
        with pytest.raises(ValueError, match="unparseable timestamps"):
            compute_lead_time(df, test_id=3)


class TestDowntimeCost:
    """The cost model prices the change in stoppage length, not the lead time."""

    def test_saving_is_independent_of_how_early_the_warning_came(self):
        """
        A longer warning does not save more money once it is long enough to act
        on. Multiplying lead time by an hourly rate would price hours the machine
        was still running normally.
        """
        m = DowntimeCostModel()
        assert m.cost_avoided_eur(30.0) == m.cost_avoided_eur(300.0)

    def test_warning_shorter_than_the_notice_period_saves_nothing(self):
        m = DowntimeCostModel(required_notice_hours=24.0)
        assert m.cost_avoided_eur(23.9) == 0.0
        assert m.cost_avoided_eur(24.0) > 0.0

    def test_no_alarm_saves_nothing(self):
        assert DowntimeCostModel().cost_avoided_eur(None) == 0.0

    def test_saving_is_the_stoppage_difference_times_the_rate(self):
        m = DowntimeCostModel(hourly_cost_eur=10_000, unplanned_hours=24, planned_hours=4)
        assert m.cost_avoided_eur(48.0) == pytest.approx(20 * 10_000)

    def test_sensitivity_spans_every_rate_given(self):
        table = cost_sensitivity(80.0, rates_eur=(5_000, 20_000))
        assert len(table) == 2
        assert table["cost_avoided_eur"].iloc[1] == 4 * table["cost_avoided_eur"].iloc[0]
        assert table["actionable"].all()

    def test_assumptions_are_rendered_for_printing_beside_any_euro_figure(self):
        text = DowntimeCostModel().assumptions()
        assert "assumptions, not measurements" in text
        assert "EUR" in text


class TestSummaryRow:
    """The summary table keeps lead time and its cost in the same record."""

    def test_summary_row_carries_lead_time_and_its_cost_together(self):
        row = compute_lead_time(make_run(), test_id=3).to_row()
        assert "lead_hours" in row
        assert "heldout_false_alarms" in row
        assert "heldout_healthy_files" in row
        assert row["shutdown_files_dropped"] == 1


# ── Published-artifact guard ───────────────────────────────────────────────


class TestPublishedSummaryGuard:
    """
    business_summary.csv is what scripts/publish.sh reads and the README quotes.

    Regression test: four single-arm runs overwrote the published three-row file
    with one arm's single row, and nothing noticed until a byte comparison.
    """

    @pytest.mark.parametrize("arm", ARMS)
    def test_an_arm_may_not_write_the_published_summary(self, arm):
        assert not writes_published_summary(arm)

    @pytest.mark.parametrize("source", ["auto", "enriched", "basic"])
    def test_the_table_backed_sources_still_write_it(self, source):
        assert writes_published_summary(source)
