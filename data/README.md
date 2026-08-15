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

## Source and retrieval

| | |
| --- | --- |
| Produced by | Center for Intelligent Maintenance Systems (IMS), University of Cincinnati, with Rexnord Technical Services |
| Distributed by | NASA Ames Prognostics Center of Excellence (PCoE) data repository |
| Dataset name | "Bearing Data Set" |
| Archive | `4. Bearings.zip` |
| Retrieved | 2026-07-15, from the PHM Society mirror |
| Extracted size | **6.2 GB** — 2.4 GB test 1, 523 MB test 2, 3.3 GB test 3 |
| File timestamps as shipped | 2020-05-16 throughout, which is the mirror's packaging date and not the acquisition date. The acquisition times are encoded in the filenames instead |

Landing pages:

- NASA PCoE: <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- PHM Society mirror: <https://data.phmsociety.org/nasa/>

Kaggle also hosts a copy. Cite the source above rather than the mirror — Kaggle did
not originate this dataset, and citing it would credit a redistributor.

## Licence and terms

**No licence file ships with the archive.** NASA PCoE distributes the data for
research use and asks that publications acknowledge both the repository and the data
donors; that request is met in the Citation section below and in the project README.

What this repository redistributes is **derived** data, never the source archive.
`data/processed/` holds six summary statistics per channel per one-second
acquisition — 27 to 51 numbers standing in for 20,480 samples per channel — which is
an aggregate, not a reconstructable copy of the waveform. The raw archive is
gitignored and is not in the history.

`data/docs/` is gitignored for a stricter reason: it holds Qiu et al. (2006),
published by Elsevier and not redistributable. Those papers are cited, never shipped.

*This section records what was done and the reasoning. It is not a legal
determination, and a reuser with different needs should check the terms at source.*

## Fetch command

Only needed to regenerate the processed tables. Run from the repository root:

```bash
mkdir -p data/raw && cd data/raw
curl -L -o bearings.zip "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
unzip bearings.zip && rm bearings.zip

# Rename to the folder names the loader expects. Note the third one: the Set 3 data
# is in a nested subfolder, NOT directly in 3rd_test/ -- see "Set 3 discrepancy".
mv 1st_test test1
mv 2nd_test test2
mv 3rd_test/4th_test/txt test3
```

Verify the counts before running anything downstream. A wrong count here is the one
error that propagates silently through every later stage:

```bash
for t in 1 2 3; do printf "test%s: %s files\n" "$t" "$(ls data/raw/test$t | wc -l)"; done
# expect  test1: 2156   test2: 984   test3: 6324
```

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

## Bearing geometry and defect frequencies

The rig runs four Rexnord ZA-2115 double-row bearings. Geometry as published in
Qiu et al. (2006):

| Symbol | Quantity | Value |
| ------ | -------- | ----- |
| `D` | pitch diameter | 2.815 in |
| `d` | ball diameter | 0.331 in |
| `n` | balls per row | 16 |
| `α` | contact angle | 15.17° |
| | shaft speed | 2,000 rpm = **33.333 Hz** |

Diameters are given in inches, but the kinematics use only the dimensionless ratio
`d/D`, so the unit cancels — provided both diameters carry the same one.

A rolling-element bearing generates a distinct repetition rate for each defect
location, because a defect is struck once per pass of a rolling element. Those rates
follow from the geometry alone (Harris, 2001) and are what makes the fault
*locatable* rather than merely detectable — energy appearing at 236 Hz and its
harmonics says outer race, not "something is wrong":

| Frequency | Name | Defect it indicates | Value |
| --------- | ---- | ------------------- | ----- |
| BPFO | ball pass frequency, outer race | outer-race spall | **236.403 Hz** |
| BPFI | ball pass frequency, inner race | inner-race spall | **296.930 Hz** |
| BSF  | ball spin frequency | rolling-element defect | **139.917 Hz** |
| FTF  | fundamental train frequency | cage damage | **14.775 Hz** |

All four are well inside the 10 kHz Nyquist limit of the 20 kHz sampling rate, so no
defect frequency is aliased.

Given the failure modes above, **BPFO carries the signal in Tests 2 and 3** and
BPFI in Test 1, with BSF relevant only to Test 1's Bearing 4 roller-element defect.
No run in this dataset fails at the cage, so FTF is never the diagnostic frequency
here.

*Provenance: `compute_defect_frequencies(BEARING_PARAMS)` in
`src/nasa_bearing_anomaly/physics.py`, with the geometry from `BEARING_PARAMS` in
`src/nasa_bearing_anomaly/config.py`, computed 2026-08-14. `tests/test_physics.py`
pins the ordering (BPFI > BPFO, FTF lowest) and BPFO to the 220–250 Hz band; the
geometry constants themselves are not asserted by any test, so they are only as
good as the Qiu et al. citation.*

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

## `data/processed/` — the committed waveform-feature tables

Regenerate with `python -m nasa_bearing_anomaly.features --test all --spectral`,
which needs `raw/` and takes about eight minutes. The summary tables above collapse
each acquisition to six statistics and discard the waveform, so nothing derived from
a spectrum can be recovered from them. These tables are the second pass that keeps
what a spectrum shows.

| File | Rows | Columns | Size |
| ---- | ---- | ------- | ---- |
| `test1_spectral.csv` | 2,156 | 144 | 2.9 MB |
| `test2_spectral.csv` |   984 |  72 | 676 kB |
| `test3_spectral.csv` | 6,324 |  72 | 4.3 MB |

Indexed by `file_index`, one row per acquisition, so it joins `test{N}_raw.csv`
row for row. Eighteen features per channel, named `{Bearing}_ch{N}_{feature}`:

| Column | Meaning |
| ------ | ------- |
| `_crest_factor` | peak / RMS, dimensionless. Impulsiveness — see the direction warning below |
| `_shape_factor` | RMS / mean absolute value |
| `_impulse_factor` | peak / mean absolute value |
| `_clearance_factor` | peak / (mean square-root amplitude)², dimensionless |
| `_energy` | sum of squares over the 20,480 samples, g² |
| `_mean_abs` | mean absolute amplitude, in g |
| `_spectral_entropy` | Shannon entropy of the normalised Welch PSD, in [0, 1]. **Higher when healthy** on all three runs |
| `_high_freq_ratio` | share of PSD power above 5 kHz, in [0, 1]. Direction is **not reliable** — see below |
| `_dominant_freq_hz` | frequency of the largest PSD bin, Hz |
| `_psd_mean`, `_psd_std` | mean and standard deviation of the PSD, g²/Hz |
| `_bpfo_energy`, `_bpfi_energy`, `_bsf_energy` | PSD power summed over three harmonic bands at that defect frequency, g²/Hz |
| `_env_kurtosis` | excess kurtosis of the envelope, dimensionless |
| `_env_bpfo_energy`, `_env_bpfi_energy`, `_env_bsf_energy` | the same three bands measured on the **envelope** spectrum |

**There is no `_ftf_energy` column, and its absence is derived rather than chosen.**
The Welch estimate uses a 0.1 s window, giving 10 Hz bins, and a band is required to
span at least three bins so that it is an integral rather than a point sample. FTF is
14.775 Hz, so any such band around it reaches 0 Hz, where the DC bin carries the
signal mean rather than defect energy. A longer window would let it back in
automatically. No test in this dataset has a cage failure, which is what FTF
diagnoses. Measured 2026-08-15: with the previous fixed 5 Hz half-width every band at
every harmonic resolved to exactly **one** bin, and `ftf_energy` separated healthy
from faulty Test 3 data by 0.02 pooled standard deviations.

**The envelope columns measure the same defect frequencies a different way, and it
is the physically correct way.** A localised defect does not radiate at its defect
frequency; each impact rings a structural resonance in the kHz range, so the defect
rate appears as the *modulation* of that resonance. The envelope columns band-pass
to 2 kHz–Nyquist, take the analytic-signal magnitude, and measure the defect bands on
the spectrum of that envelope. The band is fixed in advance rather than fitted.

**Two direction warnings, both measured by `notebooks/02_feature_engineering.ipynb`**
from a healthy window to the last 30 live acquisitions of each run:

| Column | Test 1 (inner race) | Test 2 (outer race) | Test 3 (outer race) |
| ------ | ------------------- | ------------------- | ------------------- |
| `_spectral_entropy` | 0.916 → 0.911 | 0.850 → 0.774 | 0.911 → 0.860 |
| `_high_freq_ratio`  | 0.504 → **0.507** | 0.128 → 0.122 | 0.318 → 0.232 |

Entropy falls with damage on all three, which is the opposite of the naive expectation
and is the documented behaviour: healthy noise is broadband and near-flat, which is
maximum entropy, and a defect concentrates energy into harmonics. The magnitude varies
enough that Test 1 is barely a movement.

`_high_freq_ratio` is documented to fall for the same reason, and on Test 1 it **rises**.
Do not rely on its direction. Test 1 is the inner-race run, and an inner-race defect is
amplitude-modulated by shaft rotation, which spreads energy into sidebands rather than
concentrating it — that would account for the difference, but nothing here tests it.

**These columns are strongly log-normal.** `_env_bpfo_energy` spans roughly three
orders of magnitude between healthy and failed on Test 3, so a linear reading is
dominated by its own tail. Anything comparing them should work in `log10`.

Deliberately **absent**, because `test{N}_raw.csv` already carries them: RMS, standard
deviation, absolute maximum, excess kurtosis and skewness. Storing them twice would
put two columns in a joined frame for one measurement. Values are written to six
significant figures, which is well beyond the precision of the inputs.

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

## The post-shutdown tail is a different length in each test

Measured 2026-08-14. Earlier notes in this project — and the usual reading of the IMS
documentation — described this as "the final file of each run". That is wrong in both
directions, and it matters because the tail is what a lead-time calculation anchors
against.

| Test | Trailing post-shutdown files | Last live file | Its RMS |
|---|---|---|---|
| 1 | **0** | 2155 | 0.594 — the run's **peak**, 3.7x baseline |
| 2 | **2** (982, 983) | 981 | 0.484 |
| 3 | 1 (6323) | 6322 | 0.759 — 11.4x baseline |

Test 1 never went quiet: it ends on its highest reading, so dropping one file there
discards the failure itself. Test 2 has two dead acquisitions, so dropping only one
leaves the analysis anchored on a stopped rig reading 0.002.

Because of this, `business.py` **measures** the tail rather than assuming its length: a
file counts as post-shutdown when *every* bearing's RMS falls below 0.2x the healthy
baseline. Judging it on the failed bearing alone would be wrong — one bearing can fall
quiet through damage, but all four go quiet only when the shaft stops.

*Provenance: `find_shutdown_tail` in `src/nasa_bearing_anomaly/business.py`, read from
the three `data/processed/test{N}_raw.csv` tables, columns `Bearing{N}_ch1_rms`.*

This discrepancy is documented in the peer-reviewed literature — see Ben Yagoub &
Ziani (2025), §3.2 and Table 10, which reports Bearing 3 transitioning from
Normal (file 4,448) to Suspect (file 6,093, 2004-04-16) to confirmed outer-race
failure (file 6,323, 2004-04-18). The `4th_test` folder naming is also noted by
Sahoo & Mohanty (2021). Neither source explains *why* the folder is named or
counted as it is; only that the shipped data runs to April 18 with 6,324 files.

## Provenance chain

Stages hand off through files, so any one can be re-run alone and its output
inspected:

    data/raw/test{N}/                       6.2 GB, gitignored, 20,480 samples/file
      -> loading.py                         one row per file, 6 statistics per channel
    data/processed/test{N}_raw.csv          COMMITTED — a fresh clone starts here
      -> features.py                        rolling windows (10/50), first differences
    data/processed/test{N}_features.csv     gitignored; regenerates in seconds
      -> detection.py                       StandardScaler -> PCA -> Isolation Forest

    data/raw/test{N}/                       the same archive, walked a second time
      -> features.py --spectral             Welch PSD and envelope, per channel
    data/processed/test{N}_spectral.csv     COMMITTED — joins _raw.csv on file_index

The second pass is separate rather than folded into the first because the summary
tables are published output: regenerating them to add columns would put the three
committed numbers at risk for no gain, and keeping them byte-identical is what makes
a time-domain-only run comparable against one with spectral features added.
    results/reports/test{N}_{method}_results.csv   gitignored, 1.6–10.2 MB each
    results/reports/business_summary.csv    COMMITTED — the published numbers
    results/figures/, figures/headline/     COMMITTED — the published figures

**`business.py` is the one deliberate exception to the file-handoff pattern.**
`run_business` re-runs detection in memory rather than reading
`test{N}_{method}_results.csv`, because those scored frames are gitignored — reading
them would make every published number depend on a file a fresh clone does not have.
The scored frames exist only to pass data from notebook 03 to notebook 04, which
recreates them when absent.

The notebooks run the same chain interactively: `01` explores `*_raw.csv`, `02`
builds `*_features.csv`, `03` scores it and writes the scored frames, `04` produces
the figures and the business numbers. Only the first arrow needs `data/raw/`.

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
