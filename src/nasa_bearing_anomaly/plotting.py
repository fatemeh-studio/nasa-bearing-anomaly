"""
All visualization utilities for the NASA Bearing Anomaly Detection project.

Plots produced:
1. raw_signal_overview    — RMS over time for all 4 bearings (all tests)
2. anomaly_overlay        — RMS + anomaly flags overlaid
3. kurtosis_evolution     — Kurtosis over time (key fault indicator)
4. pca_feature_space      — 2D PCA of features colored by anomaly score
5. score_distribution     — Histogram of anomaly scores (normal vs. anomaly)
6. defect_freq_heatmap    — Energy at BPFO/BPFI over time
7. summary_dashboard      — 2×3 grid for a single test (portfolio piece)

Usage:
    from nasa_bearing_anomaly.plotting import BearingPlotter
    plotter = BearingPlotter(test_id=1)
    plotter.plot_rms_overview(df)
    plotter.plot_anomaly_overlay(df_result)
    plotter.plot_summary_dashboard(df_result)
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import FIGURES_DIR, TEST_CONFIG

# ─── Style Setup ───────────────────────────────────────────────────────────

# Industrial / dark engineering aesthetic
STYLE = {
    "bg_color": "#0d1117",
    "panel_color": "#161b22",
    "text_color": "#e6edf3",
    "grid_color": "#30363d",
    "healthy_color": "#3fb950",  # green
    "anomaly_color": "#f85149",  # red
    "accent_color": "#58a6ff",  # blue
    "warn_color": "#d29922",  # amber
    "bearing_colors": ["#58a6ff", "#3fb950", "#d29922", "#bc8cff"],
}

plt.rcParams.update(
    {
        "font.family": "monospace",
        "axes.facecolor": STYLE["panel_color"],
        "figure.facecolor": STYLE["bg_color"],
        "text.color": STYLE["text_color"],
        "axes.labelcolor": STYLE["text_color"],
        "xtick.color": STYLE["text_color"],
        "ytick.color": STYLE["text_color"],
        "axes.edgecolor": STYLE["grid_color"],
        "grid.color": STYLE["grid_color"],
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
    }
)


class BearingPlotter:
    """
    Visualization suite for bearing health monitoring results.

    Parameters
    ----------
    test_id : int (1, 2, or 3)
    save_figures : bool
        If True, save PNGs to results/figures/
    """

    def __init__(self, test_id: int, save_figures: bool = True):
        self.test_id = test_id
        self.save_figures = save_figures
        self.fig_dir = FIGURES_DIR / f"test{test_id}"
        if save_figures:
            self.fig_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name: str):
        if self.save_figures:
            path = self.fig_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg_color"])
            print(f"  💾 Saved: {path}")

    # ── Plot 1: RMS Overview ───────────────────────────────────────────────

    def plot_rms_overview(self, df: pd.DataFrame) -> plt.Figure:
        """
        Show the RMS trend for all four bearings over the whole run.

        One panel per bearing, shared x-axis, so the reader can see which bearing
        departs from baseline and when.
        """
        bearings = ["Bearing1", "Bearing2", "Bearing3", "Bearing4"]
        rms_cols = {b: f"{b}_ch1_rms" for b in bearings}

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f"Test {self.test_id} — RMS Vibration Over Time (All Bearings)",
            fontsize=14,
            fontweight="bold",
            color=STYLE["text_color"],
            y=0.98,
        )

        for ax, (bearing, col), color in zip(
            axes, rms_cols.items(), STYLE["bearing_colors"], strict=True
        ):
            if col in df.columns:
                ax.plot(df.index, df[col], color=color, linewidth=0.8, alpha=0.9)
                ax.fill_between(df.index, df[col], alpha=0.15, color=color)

                # Highlight final 10% (failure zone)
                cutoff = int(len(df) * 0.90)
                ax.axvspan(cutoff, len(df), alpha=0.12, color=STYLE["anomaly_color"])
                ax.text(
                    cutoff + 5,
                    ax.get_ylim()[1] * 0.85,
                    "Failure zone",
                    color=STYLE["anomaly_color"],
                    fontsize=8,
                )

            ax.set_ylabel(f"{bearing}\nRMS (g)", fontsize=8)
            ax.grid(True, alpha=0.4)
            ax.set_xlim(0, len(df))

        axes[-1].set_xlabel("File Index (Time →)", fontsize=10)
        plt.tight_layout()

        self._save(fig, "01_rms_overview")
        return fig

    # ── Plot 2: Anomaly Overlay ────────────────────────────────────────────

    def plot_anomaly_overlay(self, df: pd.DataFrame, target_bearing: str = None) -> plt.Figure:
        """
        Plot RMS of the target bearing with flagged files highlighted.

        Defaults to the ``failed_bearing`` of the test under analysis. Note the
        markers show the raw per-file ``is_anomaly`` flag, not the sustained
        alarm that ``business.py`` derives -- the two differ, and the sustained
        rule is the one a lead time may be read from.
        """
        if target_bearing is None:
            target_bearing = TEST_CONFIG[self.test_id]["failed_bearing"]

        col = f"{target_bearing}_ch1_rms"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(
            f"Test {self.test_id} — Anomaly Detection on {target_bearing}",
            fontsize=13,
            fontweight="bold",
            color=STYLE["text_color"],
        )

        # Panel 1: RMS signal
        if col in df.columns:
            ax1.plot(df.index, df[col], color=STYLE["accent_color"], linewidth=0.8, label="RMS")

            if "is_anomaly" in df.columns:
                anom_mask = df["is_anomaly"]
                ax1.scatter(
                    df.index[anom_mask],
                    df[col][anom_mask],
                    color=STYLE["anomaly_color"],
                    s=8,
                    zorder=5,
                    label="Anomaly",
                    alpha=0.7,
                )

        ax1.set_ylabel("RMS Acceleration (g)")
        ax1.legend(loc="upper left")
        ax1.grid(True)

        # Panel 2: Anomaly score
        if "anomaly_score" in df.columns:
            score = df["anomaly_score"]
            # Normalize to [0, 1] for display
            score_norm = (score - score.min()) / (score.max() - score.min() + 1e-12)
            ax2.plot(
                df.index,
                score_norm,
                color=STYLE["warn_color"],
                linewidth=0.8,
                label="Anomaly Score",
            )

            if "is_anomaly" in df.columns:
                ax2.fill_between(
                    df.index,
                    score_norm,
                    where=df["is_anomaly"],
                    color=STYLE["anomaly_color"],
                    alpha=0.4,
                    label="Flagged",
                )

        ax2.set_ylabel("Normalized Anomaly Score")
        ax2.set_xlabel("File Index (Time →)")
        ax2.legend(loc="upper left")
        ax2.grid(True)

        plt.tight_layout()
        self._save(fig, "02_anomaly_overlay")
        return fig

    # ── Plot 3: Kurtosis Evolution ─────────────────────────────────────────

    def plot_kurtosis_evolution(self, df: pd.DataFrame) -> plt.Figure:
        """
        Kurtosis over time for all bearings.

        The `_kurt` columns hold EXCESS kurtosis, so Gaussian noise reads 0, not 3.
        The bearing literature normally quotes non-excess kurtosis — healthy 3,
        early fault 5, severe 10 — so every published threshold is 3 too high for
        this column and the reference lines below are those values minus 3.

        Physics: a healthy bearing is broadband noise, which is near-Gaussian and
        so near-zero in excess kurtosis. A spalled race strikes once per rolling
        element passing the defect, and those impulses are exactly what kurtosis
        is sensitive to.
        """
        bearings = ["Bearing1", "Bearing2", "Bearing3", "Bearing4"]
        kurt_cols = {b: f"{b}_ch1_kurt" for b in bearings}

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.set_facecolor(STYLE["panel_color"])
        fig.patch.set_facecolor(STYLE["bg_color"])

        for (bearing, col), color in zip(kurt_cols.items(), STYLE["bearing_colors"], strict=True):
            if col in df.columns:
                ax.plot(df.index, df[col], color=color, linewidth=0.9, label=bearing, alpha=0.85)

        # Reference lines, in excess kurtosis: the usual 3 / 5 / 10 thresholds
        # minus 3, because these columns already subtract the Gaussian value.
        ax.axhline(
            y=0.0,
            color=STYLE["healthy_color"],
            linestyle="--",
            alpha=0.5,
            linewidth=1.2,
            label="Gaussian baseline (excess = 0)",
        )
        ax.axhline(
            y=2.0,
            color=STYLE["warn_color"],
            linestyle="--",
            alpha=0.5,
            linewidth=1.2,
            label="Early fault (excess = 2)",
        )
        ax.axhline(
            y=7.0,
            color=STYLE["anomaly_color"],
            linestyle="--",
            alpha=0.5,
            linewidth=1.2,
            label="Severe fault (excess = 7)",
        )

        ax.set_xlabel("File Index (Time →)", fontsize=10)
        ax.set_ylabel("Kurtosis", fontsize=10)
        ax.set_title(
            f"Test {self.test_id} — Kurtosis Evolution (Fault Indicator)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True)
        ax.set_xlim(0, len(df))

        plt.tight_layout()
        self._save(fig, "03_kurtosis_evolution")
        return fig

    # ── Plot 4: PCA Feature Space ──────────────────────────────────────────

    def plot_pca_feature_space(self, df: pd.DataFrame, feature_cols: list) -> plt.Figure:
        """
        Project the feature space to 2D with PCA, coloured by anomaly score.

        Shows how far the flagged points sit from the healthy cluster. Two
        components typically capture only part of the variance, so separation
        here is indicative rather than the basis of any claim.
        """
        X = df[feature_cols].fillna(0).to_numpy()
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(10, 7))

        # Color by anomaly score
        if "anomaly_score" in df.columns:
            scores = df["anomaly_score"].to_numpy()
            # Invert: lower score = more anomalous in IsolationForest
            scatter = ax.scatter(
                X_2d[:, 0], X_2d[:, 1], c=scores, cmap="RdYlGn", s=15, alpha=0.7, edgecolors="none"
            )
            plt.colorbar(scatter, ax=ax, label="Anomaly Score (green=normal, red=anomaly)")
        else:
            ax.scatter(X_2d[:, 0], X_2d[:, 1], color=STYLE["accent_color"], s=15, alpha=0.5)

        var1, var2 = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=10)
        ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=10)
        ax.set_title(
            f"Test {self.test_id} — Feature Space (PCA Projection)\n"
            f"Normal samples form tight cluster; anomalies separate outward",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True)

        plt.tight_layout()
        self._save(fig, "04_pca_feature_space")
        return fig

    # ── Plot 5: Score Distribution ─────────────────────────────────────────

    def plot_score_distribution(self, df: pd.DataFrame) -> plt.Figure:
        """Histogram of anomaly scores: normal vs anomaly distribution."""
        if "anomaly_score" not in df.columns:
            print("  ⚠ No anomaly_score column found.")
            return None

        fig, ax = plt.subplots(figsize=(10, 5))

        normal = df[~df["is_anomaly"]]["anomaly_score"]
        anomaly = df[df["is_anomaly"]]["anomaly_score"]

        ax.hist(
            normal,
            bins=50,
            color=STYLE["healthy_color"],
            alpha=0.7,
            label=f"Normal (n={len(normal)})",
            density=True,
        )
        ax.hist(
            anomaly,
            bins=50,
            color=STYLE["anomaly_color"],
            alpha=0.7,
            label=f"Anomaly (n={len(anomaly)})",
            density=True,
        )

        ax.set_xlabel("Anomaly Score", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(
            f"Test {self.test_id} — Anomaly Score Distribution", fontsize=12, fontweight="bold"
        )
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        self._save(fig, "05_score_distribution")
        return fig

    # ── Plot 6: Summary Dashboard ──────────────────────────────────────────

    def plot_summary_dashboard(self, df: pd.DataFrame, feature_cols: list = None) -> plt.Figure:
        """
        Combine the four key panels into a single-figure dashboard.

        The centrepiece figure for the README and the site.
        """
        config = TEST_CONFIG[self.test_id]
        target = config["failed_bearing"]

        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor(STYLE["bg_color"])

        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, :])  # Full width: RMS overview
        ax2 = fig.add_subplot(gs[1, 0])  # Kurtosis
        ax3 = fig.add_subplot(gs[1, 1])  # Anomaly score
        ax4 = fig.add_subplot(gs[2, 0])  # Score distribution
        ax5 = fig.add_subplot(gs[2, 1])  # PCA (if feature_cols given)

        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.set_facecolor(STYLE["panel_color"])

        # Title
        fig.suptitle(
            f"NASA IMS Bearing Dataset — Test {self.test_id} | "
            f"{target} {config['failure_mode']} | Industry 4.0 Predictive Maintenance",
            fontsize=12,
            fontweight="bold",
            color=STYLE["text_color"],
            y=0.98,
        )

        # ── Panel 1: RMS of failed bearing ──
        col = f"{target}_ch1_rms"
        if col in df.columns:
            ax1.plot(df.index, df[col], color=STYLE["accent_color"], linewidth=0.8, label="RMS")
            if "is_anomaly" in df.columns:
                anom_mask = df["is_anomaly"]
                ax1.scatter(
                    df.index[anom_mask],
                    df[col][anom_mask],
                    color=STYLE["anomaly_color"],
                    s=6,
                    zorder=5,
                    label="Detected Anomaly",
                )
                # Lead time annotation
                first_anom = df.index[anom_mask].min() if anom_mask.any() else len(df)
                ax1.axvline(x=first_anom, color=STYLE["warn_color"], linestyle="--", linewidth=1.5)
                ax1.text(
                    first_anom + 5,
                    ax1.get_ylim()[1] * 0.7,
                    f"First alert\n(file {first_anom})",
                    color=STYLE["warn_color"],
                    fontsize=8,
                )

        ax1.set_title(f"{target} — RMS Vibration Signal", fontsize=10, pad=8)
        ax1.set_xlabel("File Index")
        ax1.set_ylabel("RMS (g)")
        ax1.legend(fontsize=8)
        ax1.grid(True)

        # ── Panel 2: Kurtosis ──
        kurt_col = f"{target}_ch1_kurt"
        if kurt_col in df.columns:
            ax2.plot(df.index, df[kurt_col], color=STYLE["warn_color"], linewidth=0.8)
            # Excess kurtosis: Gaussian is 0. See plot_kurtosis_evolution.
            ax2.axhline(
                y=0,
                color=STYLE["healthy_color"],
                linestyle="--",
                linewidth=1,
                alpha=0.6,
                label="Gaussian (0)",
            )
            ax2.axhline(
                y=2,
                color=STYLE["anomaly_color"],
                linestyle="--",
                linewidth=1,
                alpha=0.6,
                label="Fault (2)",
            )
            ax2.legend(fontsize=7)

        ax2.set_title("Kurtosis Evolution", fontsize=10)
        ax2.set_xlabel("File Index")
        ax2.set_ylabel("Kurtosis")
        ax2.grid(True)

        # ── Panel 3: Anomaly Score ──
        if "anomaly_score" in df.columns:
            score = df["anomaly_score"]
            score_norm = (score - score.min()) / (score.max() - score.min() + 1e-12)
            ax3.plot(df.index, score_norm, color=STYLE["accent_color"], linewidth=0.7)
            if "is_anomaly" in df.columns:
                ax3.fill_between(
                    df.index,
                    score_norm,
                    where=df["is_anomaly"],
                    color=STYLE["anomaly_color"],
                    alpha=0.5,
                    label="Anomaly",
                )
            ax3.legend(fontsize=8)

        ax3.set_title("Anomaly Score (normalized)", fontsize=10)
        ax3.set_xlabel("File Index")
        ax3.set_ylabel("Score")
        ax3.grid(True)

        # ── Panel 4: Score Distribution ──
        if "anomaly_score" in df.columns and "is_anomaly" in df.columns:
            n_df = df[~df["is_anomaly"]]["anomaly_score"]
            a_df = df[df["is_anomaly"]]["anomaly_score"]
            ax4.hist(
                n_df,
                bins=40,
                color=STYLE["healthy_color"],
                alpha=0.75,
                density=True,
                label="Normal",
            )
            ax4.hist(
                a_df,
                bins=40,
                color=STYLE["anomaly_color"],
                alpha=0.75,
                density=True,
                label="Anomaly",
            )
            ax4.legend(fontsize=8)

        ax4.set_title("Score Distribution", fontsize=10)
        ax4.set_xlabel("Anomaly Score")
        ax4.set_ylabel("Density")
        ax4.grid(True)

        # ── Panel 5: PCA ──
        if feature_cols and len(feature_cols) >= 2:
            X = df[feature_cols].fillna(0).to_numpy()
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)

            colors = [
                STYLE["anomaly_color"] if a else STYLE["healthy_color"]
                for a in df.get("is_anomaly", [False] * len(df))
            ]
            ax5.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=8, alpha=0.5)
            var1, var2 = pca.explained_variance_ratio_ * 100

            # Custom legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=STYLE["healthy_color"], label="Normal"),
                Patch(facecolor=STYLE["anomaly_color"], label="Anomaly"),
            ]
            ax5.legend(handles=legend_elements, fontsize=8)
            ax5.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=9)
            ax5.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=9)

        ax5.set_title("PCA Feature Space", fontsize=10)
        ax5.grid(True)

        self._save(fig, "00_summary_dashboard")
        return fig


# ─── Convenience: Plot all tests ───────────────────────────────────────────


def plot_all_tests(results: dict):
    """
    Generate and save every figure for each scored test.

    Parameters
    ----------
    results : dict
        Mapping of ``test_id`` to its scored DataFrame.
    """
    for test_id, df in results.items():
        print(f"\n📊 Plotting Test {test_id}...")
        plotter = BearingPlotter(test_id=test_id, save_figures=True)
        plotter.plot_rms_overview(df)
        plotter.plot_anomaly_overlay(df)
        plotter.plot_kurtosis_evolution(df)
        plotter.plot_score_distribution(df)
