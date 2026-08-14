"""
Failure lead time and downtime cost, from a scored detection frame.

Produces the two numbers a maintenance planner can act on: how much warning the
detector gives before the machine dies, and how often it raises an alarm on a
machine that is fine.

Why this module exists
======================
The ``is_anomaly`` column that ``detection.py`` produces is not an alarm, and
using it as one gives a lead time equal to the length of the run. That is not a
bug to be fixed downstream -- it follows from how the detector is configured.

``IsolationForest(contamination=c)`` labels a fraction ``c`` of its *training*
set anomalous by construction, and the training set here is the first 60% of the
series, the part assumed healthy. Measured on this dataset, the flagged fraction
inside the training window is 5.03% / 5.08% / 8.01% for tests 1-3 against
configured values of 0.05 / 0.05 / 0.08 -- it reproduces the parameter, because
that is what the parameter means. So a handful of the earliest files are flagged
however healthy they are, and any metric that triggers on the first flag reports
the length of the run.

A sustained-alert rule -- k of the last m windows flagged -- is necessary but on
its own is not enough. Requiring ten consecutive flags still triggers at file 9
of tests 2 and 3, because a 5-8% positive rate on healthy data produces long
runs of flags by chance. Smoothing cannot repair a label whose healthy-state
positive rate is pinned at the contamination value.

So this module does not threshold the label. It thresholds the continuous
``anomaly_score``, with the threshold calibrated on a healthy window to a chosen
false-alarm rate, and only then applies the k-of-m rule. The false-alarm rate is
an input to the calibration, not a statistic computed afterwards.

How k and m are chosen
======================
Picking k and m by looking at which values give a flattering lead time is
exactly the failure the sustained rule exists to prevent, so the choice never
looks at lead time. The grid is fixed in advance (``KM_GRID``), and the rule
selected is the first entry -- ordered by window length, then by strictness --
that raises zero sustained alarms anywhere in the healthy region it is judged on.

That criterion would be circular if the same files were then used to report the
false-alarm rate, so the healthy region is split into three disjoint parts:

===========  ===============================  ==============================
window       fraction of live series          used for
===========  ===============================  ==============================
calibration  ``[0, CALIB_FRAC)``              setting the score threshold
selection    ``[CALIB_FRAC, SELECT_FRAC)``    choosing k and m
held-out     ``[SELECT_FRAC, HEALTHY_FRAC)``  reporting the false-alarm rate
===========  ===============================  ==============================

k and m are judged on calibration and selection together; only the held-out
window is withheld. The reported rate is therefore out-of-sample with respect to
both the threshold and the rule.

The post-shutdown tail
======================
The final acquisitions of a run were recorded after the rig auto-terminated and
read near-zero root mean square (RMS). They are not the failure moment and must
not anchor a lead time. The tail is **not one file per run** -- measured on this
dataset it is 0 files for test 1, 2 for test 2 and 1 for test 3, and test 1's
last file is in fact its RMS peak. ``find_shutdown_tail`` therefore measures the
tail rather than assuming its length.

Failure is anchored on the last live acquisition: the rig was run to
destruction, so the end of live data is the failure. Lead time is the interval
from the first sustained alarm to that anchor.
"""

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, TEST_CONFIG

# ─── Tunable Constants ─────────────────────────────────────────────────────
# Fixed in advance and stated here rather than passed at the call site, so that
# a run cannot be tuned into a better answer by varying them per test.

SHUTDOWN_RMS_RATIO = 0.2
"""A file is post-shutdown when every channel's RMS falls below this multiple of
the healthy baseline. The rig drops to ~0.004 against baselines of 0.066-0.159,
so anything between about 0.05 and 0.5 separates the two populations cleanly."""

BASELINE_FRAC = 0.2
"""Fraction of the series whose median RMS defines the healthy baseline."""

CALIB_FRAC = 0.10
SELECT_FRAC = 0.25
HEALTHY_FRAC = 0.40
"""Boundaries of the three disjoint healthy windows, as fractions of the live
series. 0.40 is comfortably ahead of the earliest degradation in any of the
three runs; ``check_healthy_region`` verifies that per test rather than assuming
it."""

HEALTHY_RMS_RATIO = 2.0
"""The healthy region is only healthy if RMS stays below this multiple of
baseline throughout it. Exceeding it means HEALTHY_FRAC reaches into
degradation and the false-alarm rate would be measured on a failing bearing."""

TARGET_FAR = 0.01
"""Per-file false-alarm rate the score threshold is calibrated to, before the
k-of-m rule is applied. The sustained rule drives the realised rate far below
this."""

KM_GRID = (
    (3, 5),
    (4, 5),
    (5, 5),
    (6, 10),
    (8, 10),
    (10, 10),
    (12, 20),
    (16, 20),
    (20, 20),
    (18, 30),
    (24, 30),
    (30, 30),
)
"""Candidate (k, m) sustained-alert rules, ordered by window length and then by
strictness. Fixed before any lead time is computed. The first entry that is
clean on the selection window wins, which is the shortest and least strict rule
that does not fire on healthy data -- the most sensitive rule meeting the
constraint, not the one with the nicest lead time."""


# ─── Post-Shutdown Tail ────────────────────────────────────────────────────


def rms_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the per-bearing RMS columns of a feature table.

    Only the primary accelerometer (``_ch1_rms``) is used, so that test 1's
    eight channels and tests 2-3's four channels give one column per bearing.

    Parameters
    ----------
    df : pandas.DataFrame
        A loaded feature table.

    Returns
    -------
    list of str
        Column names ending in ``_ch1_rms``, in table order.
    """
    return [c for c in df.columns if c.endswith("_ch1_rms")]


def find_shutdown_tail(
    df: pd.DataFrame,
    baseline_frac: float = BASELINE_FRAC,
    ratio: float = SHUTDOWN_RMS_RATIO,
) -> int:
    """
    Count the trailing acquisitions recorded after the rig stopped.

    The rig auto-terminated at the end of each run, and the files written after
    it stopped carry near-zero RMS on every channel. They look like the
    healthiest readings in the series while actually being the aftermath of the
    failure, so they must be dropped before anything is anchored on the end of
    the run.

    The count is measured, not assumed. On this dataset it is 0 for test 1, 2
    for test 2 and 1 for test 3 -- the documentation's "the final file of each
    run" is wrong in both directions, and dropping exactly one file would
    discard test 1's RMS peak while leaving test 2 anchored on a dead rig.

    Shutdown is judged on the maximum across bearings rather than on the failed
    bearing alone: one bearing can fall quiet through damage, but all four go
    quiet only when the shaft stops.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature table with at least one ``_ch1_rms`` column.
    baseline_frac : float, optional
        Leading fraction of the series whose median defines the baseline.
    ratio : float, optional
        Multiple of baseline below which a file counts as post-shutdown.

    Returns
    -------
    int
        Number of trailing files to drop. Zero when the run ends live.

    Raises
    ------
    ValueError
        If the table has no ``_ch1_rms`` column.
    """
    cols = rms_columns(df)
    if not cols:
        raise ValueError(
            "No '_ch1_rms' columns found; cannot locate the post-shutdown tail. "
            f"Columns present: {list(df.columns)[:8]}..."
        )

    per_file_max = df[cols].max(axis=1)
    n_baseline = max(int(len(df) * baseline_frac), 1)
    baseline = float(per_file_max.iloc[:n_baseline].median())
    cutoff = ratio * baseline

    n_tail = 0
    for value in reversed(per_file_max.to_numpy()):
        if value < cutoff:
            n_tail += 1
        else:
            break
    return n_tail


def drop_shutdown_tail(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove the post-shutdown tail from a scored frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Scored feature table.

    Returns
    -------
    tuple of (pandas.DataFrame, int)
        The live portion of the series, and how many files were dropped.
    """
    n_tail = find_shutdown_tail(df)
    live = df.iloc[: len(df) - n_tail] if n_tail else df
    return live, n_tail


# ─── Healthy Windows ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class HealthyWindows:
    """
    The three disjoint healthy windows, as positional slices of the live series.

    Attributes
    ----------
    calibration : slice
        Files the score threshold is calibrated on.
    selection : slice
        Files k and m are chosen on.
    held_out : slice
        Files the false-alarm rate is reported on. Disjoint from both of the
        above, which is what makes the reported rate out-of-sample.
    """

    calibration: slice
    selection: slice
    held_out: slice


def healthy_windows(n_live: int) -> HealthyWindows:
    """
    Split the healthy region of a run into calibration, selection and held-out.

    Parameters
    ----------
    n_live : int
        Number of live (pre-shutdown) acquisitions.

    Returns
    -------
    HealthyWindows
        The three slices.

    Raises
    ------
    ValueError
        If the series is too short for the three windows to be non-empty.
    """
    a = int(n_live * CALIB_FRAC)
    b = int(n_live * SELECT_FRAC)
    c = int(n_live * HEALTHY_FRAC)
    if a < 1 or b - a < 1 or c - b < 1:
        raise ValueError(
            f"Series of {n_live} live files is too short to split into "
            f"calibration/selection/held-out windows at "
            f"{CALIB_FRAC}/{SELECT_FRAC}/{HEALTHY_FRAC}."
        )
    return HealthyWindows(slice(0, a), slice(a, b), slice(b, c))


def check_healthy_region(df: pd.DataFrame, failed_bearing: str) -> tuple[bool, float]:
    """
    Verify that the assumed-healthy region really is healthy.

    ``HEALTHY_FRAC`` is a fixed fraction, so on a run that degrades early it
    could reach into the failure and the false-alarm rate would then be measured
    on a bearing that is already breaking -- reporting a real detection as a
    false positive. This turns that assumption into a check.

    Parameters
    ----------
    df : pandas.DataFrame
        Live portion of the series.
    failed_bearing : str
        Bearing prefix from ``TEST_CONFIG``, e.g. ``'Bearing3'``.

    Returns
    -------
    tuple of (bool, float)
        Whether the region stayed healthy, and the peak RMS ratio observed in it.
    """
    col = f"{failed_bearing}_ch1_rms"
    series = df[col] if col in df.columns else df[rms_columns(df)].max(axis=1)

    n_baseline = max(int(len(df) * BASELINE_FRAC), 1)
    baseline = float(series.iloc[:n_baseline].median())
    end = int(len(df) * HEALTHY_FRAC)
    peak_ratio = float(series.iloc[:end].max() / baseline) if baseline > 0 else float("inf")
    return peak_ratio <= HEALTHY_RMS_RATIO, peak_ratio


# ─── Sustained-Alert Rule ──────────────────────────────────────────────────


def calibrate_threshold(
    scores: np.ndarray,
    target_far: float = TARGET_FAR,
    higher_is_anomalous: bool = False,
) -> float:
    """
    Set the score threshold from healthy data at a chosen false-alarm rate.

    ``IsolationForest.decision_function`` returns lower values for more
    anomalous points, so the threshold is the ``target_far`` quantile of the
    healthy score distribution: by construction that fraction of healthy files
    falls below it. This is the step that replaces ``contamination`` as the thing
    setting the alarm rate, and it is why the false-alarm rate is an input here
    rather than a statistic computed at the end.

    The two detectors in this project disagree about sign.
    ``IsolationForest.decision_function`` is *lower* when more anomalous, while
    the autoencoder's score is a reconstruction error and so is *higher*. Pointing
    this function at autoencoder scores without setting ``higher_is_anomalous``
    would calibrate against the wrong tail and flag the healthiest files.

    Parameters
    ----------
    scores : numpy.ndarray
        Anomaly scores from the calibration window only.
    target_far : float, optional
        Desired per-file false-alarm rate before the k-of-m rule is applied.
    higher_is_anomalous : bool, optional
        Set for reconstruction-error scores, where large means anomalous.

    Returns
    -------
    float
        Score beyond which a file is flagged -- below it when
        ``higher_is_anomalous`` is False, above it when True.
    """
    q = 1.0 - target_far if higher_is_anomalous else target_far
    return float(np.quantile(scores, q))


def flag_files(
    scores: np.ndarray, threshold: float, higher_is_anomalous: bool = False
) -> np.ndarray:
    """
    Apply a calibrated threshold with the correct comparison for the detector.

    Parameters
    ----------
    scores : numpy.ndarray
        Anomaly scores for the whole live series.
    threshold : float
        Threshold from :func:`calibrate_threshold`.
    higher_is_anomalous : bool, optional
        Set for reconstruction-error scores.

    Returns
    -------
    numpy.ndarray
        Boolean per-file flags.
    """
    return scores > threshold if higher_is_anomalous else scores < threshold


def sustained_alarm(flagged: np.ndarray, k: int, m: int) -> np.ndarray:
    """
    Apply a k-of-last-m sustained-alert rule to a per-file flag series.

    A single flagged window is noise; a fault does not go away. Requiring k of
    the last m windows is preferred to a longer moving average because it
    tolerates gaps -- an intermittent fault that flags 8 of 10 windows is real,
    and averaging would dilute it against the 2 clean ones.

    The first ``m - 1`` positions can never satisfy the rule and are False.

    Parameters
    ----------
    flagged : numpy.ndarray
        Boolean per-file flags.
    k : int
        Flags required within the window.
    m : int
        Window length in files.

    Returns
    -------
    numpy.ndarray
        Boolean array, True where the sustained rule holds.
    """
    counts = pd.Series(flagged.astype(float)).rolling(m).sum()
    return (counts >= k).to_numpy()


def select_km(
    flagged: np.ndarray, windows: HealthyWindows, grid: tuple = KM_GRID
) -> tuple[int, int] | None:
    """
    Choose the sustained-alert rule without ever looking at lead time.

    Returns the first entry of ``grid`` that raises no sustained alarm anywhere
    in the calibration *and* selection windows together. Because the grid is
    ordered by window length and then strictness, that is the shortest, least
    strict rule which is clean on healthy data -- the most sensitive rule meeting
    the constraint.

    The criterion deliberately spans both windows rather than the selection
    window alone. Test 3 opens with a burst of anomalous scores around file 28,
    inside the calibration window; judging only on the selection window left that
    burst untested, 3-of-5 was accepted, and the first sustained alarm then landed
    at file 28 for an untrustworthy lead time of 1068 h -- essentially the length
    of the run, which is the failure this whole module exists to avoid.

    The held-out window stays excluded, because that is what keeps the reported
    false-alarm rate out-of-sample: including it would force the reported rate to
    zero by construction.

    Parameters
    ----------
    flagged : numpy.ndarray
        Boolean per-file flags for the whole live series.
    windows : HealthyWindows
        The three healthy windows.
    grid : tuple, optional
        Candidate ``(k, m)`` pairs, ordered by preference.

    Returns
    -------
    tuple of (int, int) or None
        The chosen rule, or None if every candidate fires on healthy data.
    """
    region = slice(0, windows.selection.stop)
    for k, m in grid:
        if not sustained_alarm(flagged, k, m)[region].any():
            return k, m
    return None


# ─── Lead Time ─────────────────────────────────────────────────────────────


@dataclass
class LeadTimeResult:
    """
    Lead time for one test, together with everything needed to judge it.

    A lead time on its own is not a result: it has to be read against the rate
    at which the same rule fires on a healthy machine, and against how much of
    the run was discarded as post-shutdown. Every field here is reported.
    """

    test_id: int
    failed_bearing: str
    feature_source: str
    n_files: int
    n_shutdown_dropped: int
    n_live: int
    threshold: float
    k: int
    m: int
    target_far: float
    alarm_file: int | None
    alarm_time: pd.Timestamp | None
    anchor_file: int
    anchor_time: pd.Timestamp
    lead_hours: float | None
    heldout_files: int
    heldout_hours: float
    heldout_alarms: int
    healthy_region_ok: bool
    healthy_peak_ratio: float
    alarm_inside_healthy: bool
    notes: list[str] = field(default_factory=list)

    @property
    def far_per_file(self) -> float:
        """Observed sustained-alarm rate per file on the held-out healthy window."""
        return self.heldout_alarms / self.heldout_files if self.heldout_files else float("nan")

    @property
    def far_statement(self) -> str:
        """
        Render the false-alarm rate with its denominator.

        Zero alarms in a finite window is an upper bound, not a rate of zero, and
        a bare "0.0%" is the kind of number that reads as a result without being
        one. The exposure is always stated, and a zero count is rendered as
        "fewer than 1 in N".
        """
        if not self.heldout_files:
            return "not measurable (held-out window empty)"
        if self.heldout_alarms == 0:
            return (
                f"0 sustained alarms in {self.heldout_files} healthy files "
                f"({self.heldout_hours:.1f} h) -- an upper bound of "
                f"< 1 in {self.heldout_files}, not a measured rate of zero"
            )
        return (
            f"{self.heldout_alarms} sustained alarms in {self.heldout_files} healthy "
            f"files ({self.heldout_hours:.1f} h) = {100 * self.far_per_file:.2f}% of files"
        )

    def to_row(self) -> dict:
        """Flatten to one record for the summary table."""
        return {
            "test": self.test_id,
            "failed_bearing": self.failed_bearing,
            "feature_source": self.feature_source,
            "files": self.n_files,
            "shutdown_files_dropped": self.n_shutdown_dropped,
            "rule_k": self.k,
            "rule_m": self.m,
            "target_far": self.target_far,
            "score_threshold": round(self.threshold, 6),
            "alarm_file": self.alarm_file,
            "alarm_time": self.alarm_time,
            "anchor_file": self.anchor_file,
            "anchor_time": self.anchor_time,
            "lead_hours": None if self.lead_hours is None else round(self.lead_hours, 1),
            "heldout_healthy_files": self.heldout_files,
            "heldout_healthy_hours": round(self.heldout_hours, 1),
            "heldout_false_alarms": self.heldout_alarms,
            "healthy_region_ok": self.healthy_region_ok,
        }


def compute_lead_time(
    df: pd.DataFrame,
    test_id: int,
    feature_source: str = "unknown",
    target_far: float = TARGET_FAR,
    higher_is_anomalous: bool = False,
) -> LeadTimeResult:
    """
    Compute failure lead time and its false-alarm cost for one scored test.

    Runs the whole chain: drop the measured post-shutdown tail, calibrate a
    score threshold on healthy files, choose a sustained-alert rule on a second
    healthy window, then report the first sustained alarm against the last live
    acquisition -- with the false-alarm rate measured on a third, held-out
    healthy window.

    Parameters
    ----------
    df : pandas.DataFrame
        Scored frame carrying ``timestamp``, ``anomaly_score`` and the
        ``_ch1_rms`` columns, indexed by ``file_index``.
    test_id : int
        1, 2 or 3.
    feature_source : str, optional
        Which feature table produced the scores; recorded for provenance.
    target_far : float, optional
        Per-file false-alarm rate to calibrate the threshold to.
    higher_is_anomalous : bool, optional
        Set when the scores are reconstruction errors (the autoencoder) rather
        than an Isolation Forest decision function. Getting this wrong
        calibrates against the wrong tail and flags the healthiest files, which
        is why ``run_business`` derives it from the method rather than leaving
        it to the caller.

    Returns
    -------
    LeadTimeResult
        Lead time and every figure needed to judge it.

    Raises
    ------
    ValueError
        If required columns are missing, or no rule in ``KM_GRID`` is clean on
        healthy data.
    """
    required = {"timestamp", "anomaly_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Scored frame is missing required columns: {sorted(missing)}")

    config = TEST_CONFIG[test_id]
    failed = config["failed_bearing"]
    notes: list[str] = []

    live, n_tail = drop_shutdown_tail(df)

    # errors="coerce" here is not a swallowed failure: it turns both bad strings
    # and pre-existing NaT into one condition, which the next line raises on. The
    # timestamps were silently NaT across every committed table once already
    # (a pathlib .stem bug hidden by a broad except), and a lead time in hours is
    # meaningless without them, so this fails loudly and names the count.
    stamps = pd.to_datetime(live["timestamp"], errors="coerce")
    if stamps.isna().any():
        raise ValueError(
            f"{int(stamps.isna().sum())} unparseable timestamps in test {test_id}; "
            "lead time cannot be computed in hours."
        )

    windows = healthy_windows(len(live))
    region_ok, peak_ratio = check_healthy_region(live, failed)
    if not region_ok:
        notes.append(
            f"Healthy region reaches {peak_ratio:.1f}x baseline RMS, above the "
            f"{HEALTHY_RMS_RATIO}x limit: degradation starts before "
            f"{HEALTHY_FRAC:.0%} of the run, so the false-alarm rate below is "
            "measured partly on a degrading bearing and is an overestimate."
        )

    scores = live["anomaly_score"].to_numpy()
    threshold = calibrate_threshold(scores[windows.calibration], target_far, higher_is_anomalous)
    flagged = flag_files(scores, threshold, higher_is_anomalous)

    km = select_km(flagged, windows)
    if km is None:
        raise ValueError(
            f"No rule in KM_GRID is free of alarms on the healthy selection window "
            f"for test {test_id}. Widen KM_GRID or lower target_far; do not pick a "
            "rule by its lead time."
        )
    k, m = km

    triggered = sustained_alarm(flagged, k, m)
    anchor_pos = len(live) - 1
    anchor_file = int(live.index[anchor_pos])
    anchor_time = stamps.iloc[anchor_pos]

    if triggered.any():
        alarm_pos = int(np.argmax(triggered))
        alarm_file = int(live.index[alarm_pos])
        alarm_time = stamps.iloc[alarm_pos]
        lead_hours = (anchor_time - alarm_time).total_seconds() / 3600.0
        inside_healthy = alarm_pos < windows.held_out.stop
        if inside_healthy:
            notes.append(
                f"First sustained alarm at position {alarm_pos} falls inside the "
                "assumed-healthy region; the lead time below is not trustworthy."
            )
    else:
        alarm_pos = None
        alarm_file = None
        alarm_time = None
        lead_hours = None
        inside_healthy = False
        notes.append("The rule never triggers: this test yields no lead time at all.")

    ho = windows.held_out
    heldout_alarms = int(triggered[ho].sum())
    heldout_files = ho.stop - ho.start
    heldout_hours = (stamps.iloc[ho.stop - 1] - stamps.iloc[ho.start]).total_seconds() / 3600.0

    if n_tail == 0:
        notes.append(
            "No post-shutdown tail: this run ends on a live acquisition, so the "
            "anchor is the last recorded file."
        )

    return LeadTimeResult(
        test_id=test_id,
        failed_bearing=failed,
        feature_source=feature_source,
        n_files=len(df),
        n_shutdown_dropped=n_tail,
        n_live=len(live),
        threshold=threshold,
        k=k,
        m=m,
        target_far=target_far,
        alarm_file=alarm_file,
        alarm_time=alarm_time,
        anchor_file=anchor_file,
        anchor_time=anchor_time,
        lead_hours=lead_hours,
        heldout_files=heldout_files,
        heldout_hours=heldout_hours,
        heldout_alarms=heldout_alarms,
        healthy_region_ok=region_ok,
        healthy_peak_ratio=peak_ratio,
        alarm_inside_healthy=inside_healthy,
        notes=notes,
    )


# ─── Downtime Cost ─────────────────────────────────────────────────────────

PUBLISHED_HOURLY_COST = {
    "large industrial facility (Senseye/Siemens 2022)": 532_000,
    "automotive plant (Senseye/Siemens 2022)": 1_300_000,
    "fast-moving consumer goods plant (Senseye/Siemens 2022)": 36_000,
}
"""Published per-hour costs of unplanned downtime, in US dollars.

Source: Senseye Predictive Maintenance, *The True Cost of Downtime 2022*
(Siemens), compiled from 72 major multinational industrial and manufacturing
companies. https://blog.siemens.com/2023/04/the-true-cost-of-downtime/

**Scope matters more than the number.** Every figure above is the cost of
stopping an entire large facility. A single bearing on a single machine is not a
whole plant, so these are an upper bound and a sanity check on order of
magnitude, not a rate to apply to this dataset. They are recorded here so that
any euro figure in this project can be read against a sourced benchmark instead
of being asserted.
"""


@dataclass(frozen=True)
class DowntimeCostModel:
    """
    What a lead time is worth, stated as a model rather than a number.

    Lead time does not save the lead time. It converts an *unplanned* stoppage
    into a *planned* one: parts ordered in advance, a maintenance window chosen,
    no diagnosis under pressure. The saving is therefore the difference between
    the two stoppage lengths, and it is realised only if the warning arrives
    early enough to act on -- a two-hour warning buys nothing if the spare takes
    a day to arrive.

    Multiplying lead time by an hourly rate, which is the intuitive move and the
    one the previous inline version made, answers a different and meaningless
    question: it prices hours the machine was still running normally.

    Every field is an assumption, not a measurement. The defaults describe one
    machine on one line in a mid-size plant and are deliberately far below the
    whole-facility figures in ``PUBLISHED_HOURLY_COST``. Replace them with plant
    figures before quoting any euro amount as a result.

    Attributes
    ----------
    hourly_cost_eur : float
        Cost of one hour of stoppage for the affected line. Assumption.
    unplanned_hours : float
        Length of an unplanned stoppage: detect, diagnose, source the part,
        repair. Assumption.
    planned_hours : float
        Length of the same repair done in a scheduled window. Assumption.
    required_notice_hours : float
        Warning needed before the saving can be realised at all. Assumption.
    """

    hourly_cost_eur: float = 15_000.0
    unplanned_hours: float = 24.0
    planned_hours: float = 4.0
    required_notice_hours: float = 24.0

    def cost_avoided_eur(self, lead_hours: float | None) -> float:
        """
        Euro saving a given lead time buys under this model.

        Parameters
        ----------
        lead_hours : float or None
            Measured lead time. ``None`` (the rule never triggered) saves
            nothing.

        Returns
        -------
        float
            Euro saving, or 0.0 when the warning is too short to act on.
        """
        if lead_hours is None or lead_hours < self.required_notice_hours:
            return 0.0
        return (self.unplanned_hours - self.planned_hours) * self.hourly_cost_eur

    def assumptions(self) -> str:
        """Render every assumption behind a euro figure, for printing beside it."""
        return (
            f"EUR {self.hourly_cost_eur:,.0f}/h line stoppage; unplanned repair "
            f"{self.unplanned_hours:.0f} h vs planned {self.planned_hours:.0f} h; "
            f"warning must exceed {self.required_notice_hours:.0f} h to be actionable. "
            f"All four are assumptions, not measurements."
        )


def cost_sensitivity(
    lead_hours: float | None,
    rates_eur: tuple = (5_000, 10_000, 15_000, 20_000),
    model: DowntimeCostModel | None = None,
) -> pd.DataFrame:
    """
    Price a lead time across a range of hourly rates rather than at one rate.

    A single euro figure implies a precision the input does not have. The rate is
    the least defensible number in the chain, so it is varied and the reader is
    shown the range.

    Parameters
    ----------
    lead_hours : float or None
        Measured lead time.
    rates_eur : tuple, optional
        Hourly stoppage costs to evaluate.
    model : DowntimeCostModel, optional
        Base model supplying the remaining assumptions.

    Returns
    -------
    pandas.DataFrame
        One row per rate, with the euro saving under this model.
    """
    base = model or DowntimeCostModel()
    rows = []
    for rate in rates_eur:
        variant = DowntimeCostModel(
            hourly_cost_eur=float(rate),
            unplanned_hours=base.unplanned_hours,
            planned_hours=base.planned_hours,
            required_notice_hours=base.required_notice_hours,
        )
        rows.append(
            {
                "hourly_cost_eur": rate,
                "lead_hours": lead_hours,
                "actionable": lead_hours is not None and lead_hours >= base.required_notice_hours,
                "cost_avoided_eur": variant.cost_avoided_eur(lead_hours),
            }
        )
    return pd.DataFrame(rows)


# ─── Reporting ─────────────────────────────────────────────────────────────


def format_result(result: LeadTimeResult) -> str:
    """
    Render one result as text, with the false-alarm rate beside the lead time.

    The two are never printed apart: a lead time without its false-alarm cost is
    not a result.

    Parameters
    ----------
    result : LeadTimeResult
        A computed result.

    Returns
    -------
    str
        Multi-line report.
    """
    lead = "never triggers" if result.lead_hours is None else f"{result.lead_hours:.1f} h"
    lines = [
        f"Test {result.test_id} -- {result.failed_bearing} "
        f"({TEST_CONFIG[result.test_id]['failure_mode']})",
        f"  Feature source     : {result.feature_source}",
        f"  Files              : {result.n_files} "
        f"({result.n_shutdown_dropped} post-shutdown dropped, {result.n_live} live)",
        f"  Rule               : {result.k} of last {result.m} windows, "
        f"score < {result.threshold:.4f} (calibrated to {result.target_far:.1%} per-file)",
        f"  First alarm        : file {result.alarm_file} at {result.alarm_time}",
        f"  Failure anchor     : file {result.anchor_file} at {result.anchor_time}",
        f"  Lead time          : {lead}",
        f"  False-alarm cost   : {result.far_statement}",
        f"  Healthy region     : peak {result.healthy_peak_ratio:.2f}x baseline "
        f"({'ok' if result.healthy_region_ok else 'SUSPECT'})",
    ]
    for note in result.notes:
        lines.append(f"  Note: {note}")
    return "\n".join(lines)


def run_business(
    test_id: int,
    method: str = "isolation_forest",
    feature_source: str = "auto",
    target_far: float = TARGET_FAR,
) -> LeadTimeResult:
    """
    Score one test through the detection pipeline and compute its lead time.

    Detection is re-run rather than read from ``results/reports/``, because those
    scored frames are gitignored: reading them would make the headline numbers
    depend on files a fresh clone does not have.

    Parameters
    ----------
    test_id : int
        1, 2 or 3.
    method : {'isolation_forest', 'autoencoder'}
        Detector to fit.
    feature_source : {'auto', 'enriched', 'basic'}
        Which feature table to score. See ``detection.resolve_feature_table``.
    target_far : float, optional
        Per-file false-alarm rate to calibrate to.

    Returns
    -------
    LeadTimeResult
        The computed result.
    """
    from .detection import resolve_feature_table, run_pipeline

    _, source_used = resolve_feature_table(test_id, feature_source)
    scored = run_pipeline(test_id, method=method, feature_source=feature_source)
    return compute_lead_time(
        scored,
        test_id,
        feature_source=source_used,
        target_far=target_far,
        # The autoencoder scores reconstruction error, so its sign is opposite to
        # the Isolation Forest's decision function. Derived here rather than
        # asked of the caller: a wrong value produces plausible nonsense, not an
        # error.
        higher_is_anomalous=(method == "autoencoder"),
    )


# ─── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Failure lead time and false-alarm rate for the NASA IMS bearing tests"
    )
    parser.add_argument("--test", type=str, default="all", help="1, 2, 3, or 'all'")
    parser.add_argument(
        "--method",
        type=str,
        default="isolation_forest",
        choices=["isolation_forest", "autoencoder"],
    )
    parser.add_argument(
        "--feature-source", type=str, default="auto", choices=["auto", "enriched", "basic"]
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=TARGET_FAR,
        help="Per-file false-alarm rate the score threshold is calibrated to",
    )
    args = parser.parse_args()

    tests = [1, 2, 3] if args.test == "all" else [int(args.test)]
    results = []
    for t in tests:
        print(f"\n{'=' * 78}")
        print(f"Test {t}")
        print("=" * 78)
        results.append(
            run_business(
                t,
                method=args.method,
                feature_source=args.feature_source,
                target_far=args.target_far,
            )
        )

    print(f"\n{'=' * 78}")
    print("LEAD TIME AND FALSE-ALARM COST")
    print("=" * 78)
    for r in results:
        print(format_result(r))
        print()

    model = DowntimeCostModel()
    print("=" * 78)
    print("WHAT THE WARNING IS WORTH")
    print("=" * 78)
    print(f"Model assumptions: {model.assumptions()}")
    print(
        "Lead time is not multiplied by an hourly rate -- that would price hours the\n"
        "machine was still running. The saving is the difference between an unplanned\n"
        "and a planned stoppage, and only if the warning is long enough to act on.\n"
    )
    cost_rows = []
    for r in results:
        table = cost_sensitivity(r.lead_hours, model=model)
        table.insert(0, "test", r.test_id)
        cost_rows.append(table)
        actionable = "yes" if bool(table["actionable"].iloc[0]) else "no"
        lead = "none" if r.lead_hours is None else f"{r.lead_hours:.1f} h"
        span = f"{table['cost_avoided_eur'].min():,.0f} - {table['cost_avoided_eur'].max():,.0f}"
        print(
            f"  Test {r.test_id}: lead {lead:>8} | actionable at "
            f"{model.required_notice_hours:.0f} h notice: {actionable:>3} | "
            f"avoided EUR {span} across EUR 5k-20k/h"
        )
    print(
        "\nFor scale, published whole-facility figures are far larger and are NOT the\n"
        "rate used above -- a single bearing is not a whole plant:"
    )
    for label, usd in PUBLISHED_HOURLY_COST.items():
        print(f"  USD {usd:>9,.0f}/h  {label}")

    out = REPORTS_DIR / "business_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.to_row() for r in results]).to_csv(out, index=False)
    print(f"\nSaved: {out}")

    cost_out = REPORTS_DIR / "business_cost_sensitivity.csv"
    pd.concat(cost_rows, ignore_index=True).to_csv(cost_out, index=False)
    print(f"Saved: {cost_out}")
