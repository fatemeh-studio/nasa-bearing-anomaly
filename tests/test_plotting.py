"""
test_plotting.py
Tests for the figure contracts in plotting.py.
Run: pytest tests/test_plotting.py -v

Importing plotting mutates plt.rcParams process-wide, which is why the package
__init__ does not re-export it. That is harmless in a test run, and no test here
draws a figure: each one asserts a precondition that fires before any drawing.
"""

import pandas as pd
import pytest

from nasa_bearing_anomaly.plotting import BearingPlotter


@pytest.fixture
def plotter():
    return BearingPlotter(test_id=3, save_figures=False)


class TestScoreDistributionContract:
    """
    Open since Phase 0: annotated ``-> plt.Figure``, returned ``None``.

    The only caller discards the result, so an unscored frame produced a missing
    figure and no other symptom -- the same shape as the other defects this
    project has found, where a plausible default stands in for an error.
    """

    def test_unscored_frame_raises(self, plotter):
        df = pd.DataFrame({"Bearing3_ch1_rms": [0.1, 0.2, 0.3]})
        with pytest.raises(ValueError, match="needs a scored frame"):
            plotter.plot_score_distribution(df)

    def test_the_error_names_every_missing_column(self, plotter):
        df = pd.DataFrame({"Bearing3_ch1_rms": [0.1, 0.2]})
        with pytest.raises(ValueError) as exc:
            plotter.plot_score_distribution(df)
        assert "anomaly_score" in str(exc.value)
        assert "is_anomaly" in str(exc.value)

    def test_a_half_scored_frame_also_raises(self, plotter):
        """
        The old guard checked only ``anomaly_score`` while the body also reads
        ``is_anomaly``, so this frame passed the guard and then raised KeyError
        from inside the plotting code.
        """
        df = pd.DataFrame({"anomaly_score": [0.1, -0.2]})
        with pytest.raises(ValueError, match="is_anomaly"):
            plotter.plot_score_distribution(df)
