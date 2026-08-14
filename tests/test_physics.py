"""
test_physics.py
Tests for the bearing defect-frequency kinematics.
Run: pytest tests/test_physics.py -v
"""

from nasa_bearing_anomaly.config import BEARING_PARAMS
from nasa_bearing_anomaly.physics import compute_defect_frequencies


class TestDefectFrequencies:
    """Verify bearing defect frequency calculations against known values."""

    def test_frequencies_are_positive(self):
        freqs = compute_defect_frequencies(BEARING_PARAMS)
        for name, val in freqs.items():
            assert val > 0, f"{name} should be positive, got {val}"

    def test_bpfi_greater_than_bpfo(self):
        """
        BPFI is always greater than BPFO for inner race vs outer race defects.
        Physics: inner race moves faster relative to the rolling elements.
        """
        freqs = compute_defect_frequencies(BEARING_PARAMS)
        assert freqs["BPFI_Hz"] > freqs["BPFO_Hz"], (
            "BPFI should be > BPFO (inner race defects occur at higher frequency)"
        )

    def test_ftf_is_lowest(self):
        """FTF (cage frequency) should be the lowest defect frequency."""
        freqs = compute_defect_frequencies(BEARING_PARAMS)
        assert freqs["FTF_Hz"] < freqs["BPFO_Hz"]
        assert freqs["FTF_Hz"] < freqs["BSF_Hz"]

    def test_ims_rig_known_values(self):
        """
        Cross-check against published values for the IMS test rig.
        Reference: Qiu et al. (2006), Journal of Sound and Vibration.
        BPFO ≈ 236 Hz at 2000 RPM.
        """
        freqs = compute_defect_frequencies(BEARING_PARAMS)
        assert 220 < freqs["BPFO_Hz"] < 250, f"BPFO should be ~236 Hz, got {freqs['BPFO_Hz']}"
