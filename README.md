# 🛠️ Industrial Anomaly Detection: Predictive Maintenance Using NASA Bearing Dataset

> **Detecting mechanical failure before it happens — saving thousands of euros per hour in downtime costs.**

---

## 📌 Project Overview

This project applies Machine Learning and Physics-based feature engineering to detect early signs of **bearing failure** in industrial machinery, using the real-world **NASA IMS Bearing Dataset** (all three test runs).

The core idea is rooted in physics: a healthy bearing oscillates in a **stable, periodic pattern**. As it degrades, the vibration signal transitions from order → chaos. We detect this transition *before* catastrophic failure occurs.

---

## 🏭 Industrial Relevance (Austria / Industry 4.0)

| Company     | Location | Use Case                     |
| ----------- | -------- | ---------------------------- |
| Magna Steyr | Graz     | Assembly line robotic arms   |
| Voestalpine | Linz     | Rolling mill bearing systems |
| AVL List    | Graz     | Engine test bench monitoring |
| Andritz AG  | Graz     | Hydro turbine bearings       |

> **Downtime cost:** A single unplanned production stop in heavy industry costs **€5,000–€20,000 per hour**. This model provides 12–48 hours of early warning.

---

## 🧠 The Physics of Bearing Failure

Bearings fail through a well-understood progression:

```text
Stage 1 (Healthy)     → Stable periodic oscillations, low RMS
Stage 2 (Early Wear)  → Slight increase in high-frequency components
Stage 3 (Advanced)    → Sidebands appear around bearing defect frequencies
Stage 4 (Failure)     → Broadband noise, chaotic vibration, thermal spike
```

**Key physical frequencies monitored:**

- **BPFO** (Ball Pass Frequency Outer race): `n/2 × RPM/60 × (1 - d/D × cos α)`
- **BPFI** (Ball Pass Frequency Inner race): `n/2 × RPM/60 × (1 + d/D × cos α)`
- **BSF** (Ball Spin Frequency): `D/2d × RPM/60 × (1 - (d/D × cos α)²)`

---

## 📁 Project Structure

```text
nasa-bearing-anomaly/
│
├── data/
│   ├── README.md               # every data fact lives here
│   ├── raw/                    # ~6.2 GB, not committed
│   └── processed/              # three summary tables, committed
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # what healthy and degraded look like
│   ├── 02_feature_engineering.ipynb    # physics-informed features
│   ├── 03_anomaly_detection.ipynb      # Isolation Forest, optional autoencoder
│   └── 04_results_visualization.ipynb  # figures and detection summary
│
├── src/
│   └── nasa_bearing_anomaly/
│       ├── config.py           # paths, acquisition parameters, TEST_CONFIG
│       ├── physics.py          # BPFO / BPFI / BSF / FTF from rig geometry
│       ├── loading.py          # raw files → one row of statistics per file
│       ├── features.py         # time-domain and Welch-PSD features
│       ├── detection.py        # Isolation Forest + optional PyTorch autoencoder
│       └── plotting.py         # figures, including the summary dashboard
│
├── results/
│   ├── figures/                # generated PNGs
│   └── reports/                # per-test detection output
│
├── tests/                      # 53 tests, no __init__.py
│
├── pyproject.toml              # single source of truth for dependencies
├── environment.yml             # conda path into the same environment
└── README.md
```

---

## 📊 Dataset: NASA IMS Bearing Dataset

**Source:** [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) — see [`data/README.md`](data/README.md) for download and layout

| Test   | Duration              | Failed Bearing        | Failure Mode                          |
| -------| --------------------- | --------------------- | ------------------------------------- |
| Test 1 | ~35 days (2156 files) | Bearing 3 / Bearing 4 | Inner race (B3) + roller element (B4) |
| Test 2 | ~7 days (984 files)   | Bearing 1             | Outer race failure                    |
| Test 3 | ~45 days (6324 files) | Bearing 3             | Outer race failure                    |

> **⚠️ Note on Test 3:** The official IMS documentation lists Set 3 as 4,448 files ending 2004-04-04, but the distributed archive actually contains **6,324 files** (in a nested `3rd_test/4th_test/txt/` folder) running to 2004-04-18. At file 4,448 the bearing is still healthy — the outer-race failure only develops in the final ~230 files. This project uses the full 6,324-file series. Full explanation and sources in [`data/README.md`](data/README.md).

**Data format:** Test 1 has 8 channels (2 accelerometers × 4 bearings); Tests 2 and 3 have 4 channels (1 per bearing). Each file = 1 second, 20,480 samples at 20 kHz.

### Download Instructions

The raw data (~6.2 GB) is not committed to this repository. See
[`data/README.md`](data/README.md) for the download link, the expected
folder layout, and important notes on the Set 3 file-count discrepancy.

**You do not need it to run the notebooks.** The three summary tables in
`data/processed/` are committed so a clone runs without the download.

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/fatemeh-studio/nasa-bearing-anomaly.git
cd nasa-bearing-anomaly

# 2. Environment — either conda:
conda env create -f environment.yml
conda activate nasa-bearing-anomaly

#    ...or pip alone, into an environment you manage yourself:
pip install -e ".[dev,notebook]"        # add ,deep for the PyTorch autoencoder

# 3. Run the notebooks in order. No raw download needed: the summary tables
#    in data/processed/ are committed.
jupyter lab notebooks/
```

To regenerate those tables from the raw archive, place it as described in
[`data/README.md`](data/README.md) and run the pipeline. Each stage writes a CSV
the next one reads:

```bash
python -m nasa_bearing_anomaly.loading   --test all
python -m nasa_bearing_anomaly.features  --test all
python -m nasa_bearing_anomaly.detection --test all --method isolation_forest

# Check the loader without reading the whole archive. Writes nothing.
python -m nasa_bearing_anomaly.loading --test 3 --max_files 50
```

Quality gates:

```bash
ruff check . && ruff format --check .
pytest
```

---

## 🛠️ Tech Stack

| Category          | Tools                                                  |
| ----------------- | ------------------------------------------------------ |
| Data Processing   | Python, pandas, NumPy                                  |
| Signal Processing | SciPy (Welch PSD)                                      |
| Machine Learning  | scikit-learn (Isolation Forest), PyTorch (autoencoder, optional) |
| Visualization     | Matplotlib, seaborn                                    |
| Tooling           | ruff, pytest, pre-commit, GitHub Actions               |
| Environment       | Python 3.11, Jupyter Lab                               |

---

## 📈 Key Results

> **Not measured yet — placeholder.** The numbers below are illustrative and no
> code in this repository produces them. They are kept only to show the shape of
> the result, and are replaced when `business.py` lands.

| Test   | Method           | Anomaly Detected | True Failure | Lead Time       |
| ------ | ---------------- | ---------------- | ------------ | --------------  |
| Test 1 | Isolation Forest | *placeholder*    | File 2156    | *placeholder*   |
| Test 2 | Isolation Forest | *placeholder*    | File 984     | *placeholder*   |
| Test 3 | Autoencoder      | *placeholder*    | File 6324    | *placeholder*   |

A lead time is not a result on its own. Three things have to accompany it, and
none of them exists yet:

- a **sustained-alert rule** — k of the last m windows flagged — because triggering
  on the first anomaly lets one early false positive inflate the number
- a **false-alarm rate** calibrated against a healthy baseline, reported alongside
- the **post-shutdown tail excluded**: the final file of each run was recorded after
  the rig had already stopped, so it is not the moment of failure

Until those land there is no downtime-cost figure to quote either.

---

## ✉️ Keywords for Industry 4.0 Recruiters

`Predictive Maintenance` · `Condition Monitoring` · `Industry 4.0` · `Anomaly Detection` · `Vibration Analysis` · `Smart Manufacturing` · `Signal Processing` · `IIoT` · `Bearing Diagnostics`

---

*J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007).*
*IMS, University of Cincinnati. "Bearing Data Set",*
*NASA Ames Prognostics Data Repository,*
*NASA Ames Research Center, Moffett Field, CA.*
*<https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip>*
