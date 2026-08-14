# Data — NASA IMS bearing vibration

Everything this project knows about its data: where it comes from, what is and is
not committed, what the processed tables contain, and one documented defect in the
source archive. Nothing else in the repository should restate these facts; it should
link here.

    data/
    ├── README.md      this file
    ├── raw/           ~6.2 GB, gitignored — download instructions below
    ├── processed/     three summary tables, committed on purpose
    └── docs/          reference PDFs, gitignored and never redistributed

**You do not need the raw data to run any of the four notebooks.** The three tables
in `data/processed/` are committed precisely so that a clone runs without a 6.2 GB
download. You need `raw/` only to regenerate those tables.

`data/docs/` holds reference papers kept beside the code while working. It is
gitignored and stays that way: one of them is published by Elsevier and may not be
redistributed. The papers are cited at the end of this file, not shipped.

## Download

Direct (PHM Society mirror — reliable):

    https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip

Landing pages:

- NASA PCoE: <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- PHM Society mirror: <https://data.phmsociety.org/nasa/>

## Expected layout

The archive extracts to `1st_test/`, `2nd_test/`, `3rd_test/` and a documentation
PDF. **Note the nested folder** (see "Set 3 discrepancy" below): the Set 3 data
actually lives at `3rd_test/4th_test/txt/`, not directly in `3rd_test/`.

Arrange the folders so the tree looks like:

    data/raw/
    ├── test1/   # 2,156 files, 8 channels (4 bearings x 2 accelerometers)
    ├── test2/   #   984 files, 4 channels (1 per bearing)
    └── test3/   # 6,324 files, 4 channels (1 per bearing)
                 #   ^ from 3rd_test/4th_test/txt/ — see note below

Each file is a 1-second vibration snapshot: 20,480 points at a 20 kHz sampling
rate (the effective rate is closer to 20.48 kHz). Rig: shaft at 2,000 RPM under a
6,000 lb radial load, four force-lubricated Rexnord ZA-2115 double-row bearings,
run to failure.

Failure at the end of each test (per the IMS documentation):

| Test | Files | Channels | Failure at end of test                          |
| ---- | ----- | -------- | ----------------------------------------------- |
| 1    | 2,156 | 8        | Bearing 3 inner race + Bearing 4 roller element |
| 2    |   984 | 4        | Bearing 1 outer race                            |
| 3    | 6,324 | 4        | Bearing 3 outer race                            |

## `data/processed/` — the committed summary tables

Regenerate with `python -m nasa_bearing_anomaly.loading --test all`, which needs
`raw/`. Each raw file — one second, 20,480 samples — collapses to **one row** of
per-channel statistics, so the temporal ordering the whole analysis rests on is
preserved while the 6.2 GB does not have to be.

| File | Rows | Columns | Size |
| ---- | ---- | ------- | ---- |
| `test1_raw.csv` | 2,156 | 51 | 1.8 MB |
| `test2_raw.csv` |   984 | 27 | 452 kB |
| `test3_raw.csv` | 6,324 | 27 | 2.9 MB |

Three index columns, then six statistics per channel named
`{Bearing}_ch{N}_{stat}`:

| Column | Meaning |
| ------ | ------- |
| `file_index` | 0-based position in the run. The table's index, and the x-axis of every figure |
| `timestamp`  | parsed from the raw filename (`YYYY.MM.DD.HH.MM.SS`). Acquisitions are 10 minutes apart |
| `filename`   | the raw file this row came from |
| `_rms`  | root mean square of the 20,480 samples, in g. The primary degradation indicator |
| `_mean` | arithmetic mean, in g. Near zero for a healthy AC-coupled accelerometer |
| `_std`  | standard deviation, in g |
| `_max`  | maximum **absolute** amplitude, in g |
| `_kurt` | **excess** kurtosis — see the warning below |
| `_skew` | skewness, dimensionless |

**`_kurt` is excess kurtosis, so Gaussian noise reads 0, not 3.** The computation
subtracts 3 (Fisher definition). This matters because the bearing-diagnostics
literature usually quotes *non-excess* kurtosis, where healthy is 3, early fault
4–6 and severe fault above 10 — apply those numbers to this column and every
threshold is 3 too high. Subtract 3 from any published threshold before using it
here. Measured on Test 3, Bearing 3: **healthy median 0.15**, final 30 files
median 3.17, maximum 16.74.

Test 1 has eight channels and so 48 statistic columns; Tests 2 and 3 have four
channels and 24. There is no `_ch2` column for Tests 2 and 3, because there was no
second accelerometer.

## Set 3 discrepancy (the `4th_test` folder and the 4,448 vs 6,324 file count)

This dataset has a well-documented quirk in its third test that is easy to get
wrong, so it is worth stating precisely.

**What the official Readme says.** The IMS "Readme Document for IMS Bearing Data"
describes Set 3 as **4,448 files**, recorded from **2004-03-04 09:27:46 to
2004-04-04 19:01:57**, ending in an outer-race failure of Bearing 3.

**What the distributed archive actually contains.** Inside `3rd_test/` there is a
nested folder `4th_test/txt/` holding **6,324 files**, recorded continuously at
10-minute intervals through to **2004-04-18 02:42:55** — roughly two weeks past
the date the Readme reports. This is the real Set 3; there is no separate "Set 4"
experiment, and no files are missing. The folder name and the Readme's file count
are simply inconsistent with the shipped data.

**Why it matters.** At file 4,448 (2004-04-04, the Readme's cutoff) Bearing 3 is
still healthy. The outer-race defect only develops and becomes unmistakable in the
final ~230 files (2004-04-16 to 2004-04-18). Analyses that trust the Readme and
stop at 4,448 files therefore **miss the failure entirely** — the run looks like it
ends in a healthy state. Using the full 6,324-file series is required to capture
the degradation this project is built to detect.

**This project uses the full 6,324-file series as `test3/`.**

**Measured confirmation.** Per-bearing RMS computed from the raw files
(channel N = Bearing N). Baseline is the median of the first 500 files,
**B3 = 0.0664**:

    File   Date / time      RMS[B1    B2    B3    B4]   B3 vs baseline
       0   2004-03-04 09:27     0.080 0.097 0.066 0.055    1.0x  healthy
    4447   2004-04-04 19:01     0.074 0.077 0.070 0.055    1.1x  still healthy — the Readme's documented end
    6092   2004-04-16 12:12     0.077 0.084 0.085 0.064    1.3x  degradation begins
    6299   2004-04-17 22:42     0.121 0.159 0.297 0.161    4.5x
    6320   2004-04-18 02:12     0.138 0.207 0.454 0.254    6.8x
    6321   2004-04-18 02:22     0.147 0.251 0.590 0.280    8.9x
    6322   2004-04-18 02:32     0.154 0.257 0.759 0.282   11.4x  maximum
    6323   2004-04-18 02:42     0.002 0.003 0.004 0.002    0.1x  post-shutdown (rig stopped)

**The failure accelerates sharply in its final half hour.** Bearing 3 reaches about
4.5x baseline by 19:30 on 2004-04-17 and holds near there for some hours, then more
than doubles across the last three acquisitions — 0.454, 0.590, 0.759 — before the rig
stops. Only 29 of the 6,324 files exceed 0.297. **Quoting 0.297 as "the failure value"
understates the peak by a factor of 2.6 and misplaces the failure point by roughly four
hours**, which matters directly for any lead-time calculation that must anchor on one.

At the Readme's 4,448-file cutoff (2004-04-04 19:01) Bearing 3 is still at 1.1x
baseline — the failure is entirely contained in the files beyond the documented range.
The other three bearings rise too, almost all of it in those same final acquisitions:
B4 reaches 5.1x and B2 2.6x. Through 22:42 their movement is mild and consistent with
sympathetic vibration through the shaft; across the last three files it is not. The
final snapshot reads near-zero on every channel because the rig had already stopped
(it auto-terminates on a debris-triggered switch), so the post-shutdown tail must be
excluded when computing failure lead-time.

*Provenance: every figure above is read from `data/processed/test3_raw.csv`, columns
`Bearing{N}_ch1_rms`, regenerated 2026-08-12 with
`python -m nasa_bearing_anomaly.loading --test 3`. File numbers are the `file_index`
column; dates are parsed from the raw filenames. All five rows present before
2026-08-12 reproduced exactly; the three tail rows and the ratio column are new.*

This discrepancy is documented in the peer-reviewed literature — see Ben Yagoub &
Ziani (2025), §3.2 and Table 10, which reports Bearing 3 transitioning from
Normal (file 4,448) to Suspect (file 6,093, 2004-04-16) to confirmed outer-race
failure (file 6,323, 2004-04-18). The `4th_test` folder naming is also noted by
Sahoo & Mohanty (2021). Neither source explains *why* the folder is named or
counted as it is; only that the shipped data runs to April 18 with 6,324 files.

## Citation

Dataset:

    J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007).
    IMS, University of Cincinnati. "Bearing Data Set", NASA Ames Prognostics
    Data Repository, NASA Ames Research Center, Moffett Field, CA.

The repository asks that publications acknowledge both the repository and the
data donors.

References documenting the Set 3 discrepancy:

    A. Ben Yagoub and R. Ziani (2025). "Bearing fault prognosis based on Cyclical
    Remaining Useful Life (CRUL)." Comptes Rendus. Mécanique 353, pp. 1477–1495.
    https://doi.org/10.5802/crmeca.332

    B. Sahoo and A. R. Mohanty (2021). "Multiclass Bearing Fault Classification
    Using Features Learned by a Deep Neural Network." International Congress and
    Workshop on Industrial AI, Springer, pp. 405–414.

Primary reference for the test rig and failure mechanism:

    H. Qiu, J. Lee, J. Lin, and G. Yu (2006). "Wavelet filter-based weak signature
    detection method and its application on rolling element bearing prognostics."
    Journal of Sound and Vibration 289 (4–5), pp. 1066–1090.
