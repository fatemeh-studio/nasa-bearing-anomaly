"""
Frozen constants: filesystem roots, test-run configuration, rig geometry.

Covers the acquisition parameters and the bearing geometry of the NASA IMS test
rig alongside the per-test configuration.

Nothing here is computed from data and nothing here has a default that another
module may override. A magic number anywhere else in the package is a bug.

Every fact about the dataset itself -- source, span, licence, file counts,
channel meanings, known defects -- lives in data/README.md. The values below are
the subset the code has to hold in memory to run at all.
"""

from pathlib import Path

# ─── Filesystem Roots ──────────────────────────────────────────────────────
# Resolved relative to this file rather than the working directory, so the
# pipeline behaves the same from the repo root, from notebooks/, and from a
# scheduled job. This assumes an editable install (`pip install -e .`), which is
# the documented install; a non-editable install would place these under
# site-packages, where no data directory exists.

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_ROOT / "figures"
REPORTS_DIR = RESULTS_ROOT / "reports"

# The one figure the README embeds, kept apart from the generated ones so the
# portfolio-wide convention holds: figures/headline/ is what a README may point
# at, results/figures/ is everything the notebooks produce.
HEADLINE_DIR = PROJECT_ROOT / "figures" / "headline"


def repo_path(path: Path) -> str:
    """
    Render a path for display, relative to the repository root.

    The constants above are absolute so that a stage behaves identically whatever
    directory the kernel started in. Printing them raw is a different matter: the
    prints are captured into notebook stored outputs, and Quarto renders those
    onto the published site, so an absolute path puts the author's home directory
    and username on a public page. It is also useless to a reader, whose checkout
    is somewhere else entirely.

    Parameters
    ----------
    path : pathlib.Path
        Any path, inside or outside the repository.

    Returns
    -------
    str
        The path relative to ``PROJECT_ROOT`` when it lies inside the repository,
        otherwise the path unchanged -- an absolute path from elsewhere is
        genuinely the useful thing to show.
    """
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ─── Acquisition Parameters ────────────────────────────────────────────────
# Nominal rate as used throughout the IMS documentation. The effective rate is
# closer to 20.48 kHz; the nominal value is what every published analysis of this
# dataset uses for its frequency axis, so it is what the defect-frequency bands
# are matched against. See data/README.md.

SAMPLING_RATE_HZ = 20_000
SAMPLES_PER_FILE = 20_480


# ─── Bearing Geometry (IMS Test Rig) ───────────────────────────────────────
# Rexnord ZA-2115 double-row bearing, as published in Qiu, Lee, Lin and Yu
# (2006), Journal of Sound and Vibration 289(4-5), pp. 1066-1090.
#
# Diameters are in inches. The kinematic equations in physics.py use only the
# ratio ball/pitch, which is dimensionless, so the choice of unit cancels --
# provided both diameters carry the same one.

BEARING_PARAMS = {
    "pitch_diameter": 2.815,  # D (inches)
    "ball_diameter": 0.331,  # d (inches)
    "num_balls": 16,  # n, per row
    "contact_angle_deg": 15.17,  # alpha
    "rpm": 2000,  # shaft speed
}


# ─── Test-Run Configuration ────────────────────────────────────────────────
# Single source of truth for paths, channel names, failed bearing and failure
# mode. The detection pipeline and the plotter both read failed_bearing from
# here to decide which columns to focus on, so editing this dict silently
# re-targets the whole pipeline.
#
# columns[j] labels physical channel j, so the list length must equal the number
# of physical channels that test recorded. load_test verifies this before reading
# anything and raises on a mismatch: without the check, a list longer than the
# data mislabels every channel instead of failing, and the analysis runs to
# completion on the wrong bearing. Channel counts per test are in data/README.md.

TEST_CONFIG = {
    1: {
        "raw_dir": DATA_ROOT / "raw" / "test1",
        "output_file": DATA_ROOT / "processed" / "test1_raw.csv",
        "columns": [
            "Bearing1_ch1",
            "Bearing1_ch2",
            "Bearing2_ch1",
            "Bearing2_ch2",
            "Bearing3_ch1",
            "Bearing3_ch2",
            "Bearing4_ch1",
            "Bearing4_ch2",
        ],
        "failed_bearing": "Bearing3",
        "failure_mode": "Inner race failure (Bearing 4 roller element also failed)",
        "total_files": 2156,
    },
    2: {
        "raw_dir": DATA_ROOT / "raw" / "test2",
        "output_file": DATA_ROOT / "processed" / "test2_raw.csv",
        "columns": ["Bearing1_ch1", "Bearing2_ch1", "Bearing3_ch1", "Bearing4_ch1"],
        "failed_bearing": "Bearing1",
        "failure_mode": "Outer race failure",
        "total_files": 984,
    },
    3: {
        "raw_dir": DATA_ROOT / "raw" / "test3",
        "output_file": DATA_ROOT / "processed" / "test3_raw.csv",
        "columns": ["Bearing1_ch1", "Bearing2_ch1", "Bearing3_ch1", "Bearing4_ch1"],
        "failed_bearing": "Bearing3",
        "failure_mode": "Outer race failure",
        "total_files": 6324,
    },
}
