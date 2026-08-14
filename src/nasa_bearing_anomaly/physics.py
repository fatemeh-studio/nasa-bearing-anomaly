"""
Bearing defect-frequency kinematics.

A rolling-element bearing with a localised defect produces an impulse every time
a rolling element passes the damaged spot. The repetition rate of those impulses
is fixed by the bearing's geometry and shaft speed alone, not by the severity of
the damage -- which is what makes it diagnostic: energy appearing at one of these
four frequencies names *which surface* has failed, long before overall vibration
amplitude rises enough to notice.

Four characteristic frequencies, all in Hz:

- **BPFO**, ball pass frequency outer race -- a defect on the stationary outer
  race, struck once per rolling element passing it.
- **BPFI**, ball pass frequency inner race -- a defect on the rotating inner
  race. Higher than BPFO because the inner race moves with the shaft, so the
  relative passing rate is greater.
- **BSF**, ball spin frequency -- a defect on a rolling element itself.
- **FTF**, fundamental train frequency -- the cage rotation rate, the slowest of
  the four.

Kept separate from file input/output because these bands are what the
frequency-domain features in features.py are measured against.

Equations from Harris (2001), *Rolling Bearing Analysis*. Rig geometry and its
citation are in config.py.
"""

import numpy as np

# ─── Defect Frequencies ────────────────────────────────────────────────────


def compute_defect_frequencies(params: dict) -> dict:
    """
    Compute characteristic bearing defect frequencies.

    Based on kinematic bearing equations (Harris, 2001).

    Returns dict with BPFO, BPFI, BSF, FTF in Hz.
    """
    n = params["num_balls"]
    d = params["ball_diameter"]
    D = params["pitch_diameter"]
    alpha = np.radians(params["contact_angle_deg"])
    rpm = params["rpm"]
    shaft_Hz = rpm / 60.0

    ratio = (d / D) * np.cos(alpha)

    BPFO = (n / 2) * shaft_Hz * (1 - ratio)
    BPFI = (n / 2) * shaft_Hz * (1 + ratio)
    BSF = (D / (2 * d)) * shaft_Hz * (1 - ratio**2)
    FTF = (shaft_Hz / 2) * (1 - ratio)

    return {
        "BPFO_Hz": round(BPFO, 3),
        "BPFI_Hz": round(BPFI, 3),
        "BSF_Hz": round(BSF, 3),
        "FTF_Hz": round(FTF, 3),
    }
