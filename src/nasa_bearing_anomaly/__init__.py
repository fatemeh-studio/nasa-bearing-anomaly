"""
Unsupervised anomaly detection on the NASA IMS bearing run-to-failure dataset.

Modules, in the order the pipeline runs:

    config      frozen constants -- paths, acquisition parameters, rig geometry,
                and TEST_CONFIG, the single source of truth for per-test setup
    physics     bearing defect-frequency kinematics (BPFO, BPFI, BSF, FTF)
    loading     raw acquisition files -> one row of summary statistics per file
    features    time-domain and Welch-PSD features from a raw waveform, plus
                rolling and difference columns over the summary table
    detection   Isolation Forest, and a PyTorch autoencoder when [deep] is
                installed
    plotting    figures, including the summary dashboard

Nothing is re-exported here. Import from the module that owns it:

    from nasa_bearing_anomaly.loading import load_test

Keeping this file empty means `import nasa_bearing_anomaly` proves the package
is installed without pulling in scikit-learn, matplotlib and a PyTorch probe --
and without matplotlib's global style being mutated as a side effect of the
import, which is what plotting does at module level.

Dataset facts -- source, span, licence, channel counts, known defects -- are in
data/README.md.
"""
