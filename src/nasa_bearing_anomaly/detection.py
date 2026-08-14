"""
Two complementary unsupervised anomaly detectors for bearing vibration.

The approaches:

1. ISOLATION FOREST (sklearn)
   - Unsupervised: no labels needed
   - Fast and explainable
   - Works well when anomalies are rare (contamination ~ 5%)
   - Good for Test 1 and Test 2

2. AUTOENCODER (PyTorch)
   - Learns to reconstruct "normal" vibration patterns
   - Anomaly score = reconstruction error
   - Better for gradual degradation (Test 3)
   - Requires more data but captures non-linear patterns

Both produce an anomaly_score column. Values above threshold
are flagged as anomalies. Threshold is tunable per test.

Usage:
    from nasa_bearing_anomaly.detection import IsolationForestDetector, AutoencoderDetector

    detector = IsolationForestDetector(contamination=0.05)
    df_result = detector.fit_predict(df, feature_cols)
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import REPORTS_DIR, TEST_CONFIG
from .loading import load_processed

# Try importing PyTorch (optional, graceful fallback)
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print(
        "⚠ PyTorch not available. AutoencoderDetector disabled. Install: pip install -e '.[deep]'"
    )


# ─── Isolation Forest Detector ─────────────────────────────────────────────


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector for bearing health monitoring.

    Physics rationale: Normal bearing vibration forms a compact cluster
    in feature space. Isolation Forest isolates outliers by randomly
    partitioning data — anomalous points require fewer partitions.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalies (0.0–0.5).
        Conservative: 0.05 (5%). Adjust per test duration.
    n_estimators : int
        Number of trees (more = more stable, slower)
    use_pca : bool
        Reduce dimensionality before detection (helps with high-dim features)
    n_components : int
        PCA components if use_pca=True
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        use_pca: bool = True,
        n_components: int = 10,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.use_pca = use_pca
        self.n_components = n_components
        self.random_state = random_state

        self.scaler = StandardScaler()
        self._pca_n_components = n_components
        self.pca = None  # initialized in fit() once we know actual n_features
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_cols_ = None

    def _prepare(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """Scale and optionally apply PCA."""
        X = df[feature_cols].fillna(0).to_numpy()
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        return X_scaled

    def fit(self, df: pd.DataFrame, feature_cols: list) -> "IsolationForestDetector":
        """
        Fit on the assumed-healthy first 60% of the time series.

        Early files are taken to represent normal operation, which holds for a
        run-to-failure rig. Note that ``contamination`` still labels that
        fraction of this training set anomalous, so the fitted model flags some
        of these healthy files by construction -- see ``business.py``, which is
        why a lead time cannot be read off ``is_anomaly`` directly.

        Parameters
        ----------
        df : pandas.DataFrame
            Full feature table, in time order.
        feature_cols : list
            Columns to fit on.

        Returns
        -------
        IsolationForestDetector
            This detector, fitted.
        """
        self.feature_cols_ = feature_cols
        n_healthy = int(len(df) * 0.60)
        df_train = df.iloc[:n_healthy]

        X = df_train[feature_cols].fillna(0).to_numpy()
        X_scaled = self.scaler.fit_transform(X)
        if self.use_pca:
            n_components = min(self._pca_n_components, X_scaled.shape[0], X_scaled.shape[1])
            self.pca = PCA(n_components=n_components)
            X_scaled = self.pca.fit_transform(X_scaled)

        self.model.fit(X_scaled)
        print(
            f"  ✅ IsolationForest fitted on {n_healthy} healthy samples ({len(feature_cols)} features)"
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomaly scores for all samples.

        Returns df with added columns:
            - anomaly_score: raw decision function (lower = more anomalous)
            - anomaly_label: 1 (normal) or -1 (anomaly)
            - is_anomaly: boolean True/False
        """
        X = self._prepare(df, self.feature_cols_)
        df = df.copy()

        # decision_function: negative = anomalous, positive = normal
        df["anomaly_score"] = self.model.decision_function(X)
        df["anomaly_label"] = self.model.predict(X)
        df["is_anomaly"] = df["anomaly_label"] == -1

        n_anomalies = df["is_anomaly"].sum()
        print(
            f"  🔴 Anomalies detected: {n_anomalies}/{len(df)} ({100 * n_anomalies / len(df):.1f}%)"
        )

        # Find first anomaly index
        first_anomaly = df[df["is_anomaly"]].index.min()
        if not pd.isna(first_anomaly):
            lead_time = len(df) - first_anomaly
            print(
                f"  ⏱  First anomaly at sample {first_anomaly} "
                f"(lead time: {lead_time} samples before end)"
            )

        return df

    def fit_predict(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Fit on the healthy portion, then score the whole series.

        Parameters
        ----------
        df : pandas.DataFrame
            Full feature table, in time order.
        feature_cols : list
            Columns to fit and score on.

        Returns
        -------
        pandas.DataFrame
            The table with the three result columns appended.
        """
        self.fit(df, feature_cols)
        return self.predict(df)

    def save(self, path: Path):
        """
        Persist scaler, PCA, model and feature list to ``path``.

        Parameters
        ----------
        path : pathlib.Path
            Directory to write ``isolation_forest.pkl`` into. Created if absent.
        """
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "pca": self.pca,
                "model": self.model,
                "feature_cols": self.feature_cols_,
            },
            path / "isolation_forest.pkl",
        )
        print(f"  💾 Model saved to {path / 'isolation_forest.pkl'}")

    @classmethod
    def load(cls, path: Path) -> "IsolationForestDetector":
        """
        Restore a detector saved by :meth:`save`.

        Parameters
        ----------
        path : pathlib.Path
            Directory holding ``isolation_forest.pkl``.

        Returns
        -------
        IsolationForestDetector
            A detector ready to ``predict``; its constructor arguments are not
            restored, only the fitted state.
        """
        data = joblib.load(path / "isolation_forest.pkl")
        obj = cls()
        obj.scaler = data["scaler"]
        obj.pca = data["pca"]
        obj.model = data["model"]
        obj.feature_cols_ = data["feature_cols"]
        return obj


# ─── Autoencoder Detector (PyTorch) ────────────────────────────────────────

if TORCH_AVAILABLE:

    class BearingAutoencoder(nn.Module):
        """
        Simple feedforward autoencoder for bearing vibration features.

        Architecture: Encoder compresses features → bottleneck → Decoder reconstructs.
        Anomaly score = Mean Squared Error of reconstruction.

        The encoder learns the manifold of NORMAL behavior.
        Anomalous samples cannot be reconstructed well → high MSE.
        """

        def __init__(self, input_dim: int, bottleneck_dim: int = 8):
            super().__init__()
            hidden = max(input_dim // 2, bottleneck_dim * 2)

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, bottleneck_dim * 2),
                nn.ReLU(),
                nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, bottleneck_dim * 2),
                nn.ReLU(),
                nn.Linear(bottleneck_dim * 2, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, input_dim),
            )

        def forward(self, x):
            """
            Encode to the bottleneck and reconstruct.

            Parameters
            ----------
            x : torch.Tensor
                Batch of scaled feature vectors.

            Returns
            -------
            torch.Tensor
                Reconstruction, same shape as ``x``.
            """
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat

    class AutoencoderDetector:
        """
        Autoencoder-based anomaly detector.

        Best suited for Test 3 where degradation is gradual and multi-mode.

        Parameters
        ----------
        bottleneck_dim : int
            Latent space dimension. Smaller = more compression = stricter "normal" definition.
        epochs : int
            Training epochs (100–300 for this dataset size)
        lr : float
            Learning rate
        threshold_percentile : float
            Reconstruction error percentile on training set used as anomaly threshold.
        """

        def __init__(
            self,
            bottleneck_dim: int = 8,
            epochs: int = 150,
            lr: float = 1e-3,
            batch_size: int = 32,
            threshold_percentile: float = 95.0,
        ):
            self.bottleneck_dim = bottleneck_dim
            self.epochs = epochs
            self.lr = lr
            self.batch_size = batch_size
            self.threshold_percentile = threshold_percentile

            self.scaler = StandardScaler()
            self.model: BearingAutoencoder | None = None
            self.threshold_: float | None = None
            self.feature_cols_ = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def fit(self, df: pd.DataFrame, feature_cols: list) -> "AutoencoderDetector":
            """
            Train on the assumed-healthy first 60%, then set the error threshold.

            The threshold is the ``threshold_percentile`` of reconstruction error
            over the training set, so unlike the Isolation Forest's
            ``contamination`` it is an explicit quantile of healthy error rather
            than a forced label rate.

            Parameters
            ----------
            df : pandas.DataFrame
                Full feature table, in time order.
            feature_cols : list
                Columns to train on.

            Returns
            -------
            AutoencoderDetector
                This detector, trained, with ``threshold_`` set.
            """
            self.feature_cols_ = feature_cols
            n_healthy = int(len(df) * 0.60)
            df_train = df.iloc[:n_healthy]

            X_train = df_train[feature_cols].fillna(0).to_numpy()
            X_scaled = self.scaler.fit_transform(X_train)

            self.model = BearingAutoencoder(
                input_dim=len(feature_cols),
                bottleneck_dim=self.bottleneck_dim,
            ).to(self.device)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()

            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            print(
                f"  🤖 Training Autoencoder ({len(feature_cols)} features → {self.bottleneck_dim}D bottleneck)..."
            )
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for x_batch, _ in loader:
                    optimizer.zero_grad()
                    x_hat = self.model(x_batch)
                    loss = criterion(x_hat, x_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 50 == 0:
                    print(
                        f"    Epoch [{epoch + 1}/{self.epochs}] Loss: {total_loss / len(loader):.6f}"
                    )

            # Compute threshold from training reconstruction errors
            self.model.eval()
            with torch.no_grad():
                x_hat_train = self.model(X_tensor)
                train_errors = torch.mean((X_tensor - x_hat_train) ** 2, dim=1).cpu().numpy()

            self.threshold_ = float(np.percentile(train_errors, self.threshold_percentile))
            print(
                f"  ✅ Threshold set at {self.threshold_percentile}th percentile: {self.threshold_:.6f}"
            )
            return self

        def predict(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Score every sample by reconstruction error.

            Note the sign convention differs from the Isolation Forest: here a
            **higher** ``anomaly_score`` is more anomalous, because the score is
            an error. ``business.py`` therefore cannot be pointed at autoencoder
            output without flipping the comparison.

            Parameters
            ----------
            df : pandas.DataFrame
                Feature table carrying the columns fitted on.

            Returns
            -------
            pandas.DataFrame
                The table with ``anomaly_score``, ``is_anomaly`` and
                ``anomaly_label`` appended.
            """
            X = df[self.feature_cols_].fillna(0).to_numpy()
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            self.model.eval()
            with torch.no_grad():
                x_hat = self.model(X_tensor)
                errors = torch.mean((X_tensor - x_hat) ** 2, dim=1).cpu().numpy()

            df = df.copy()
            df["anomaly_score"] = errors
            df["is_anomaly"] = errors > self.threshold_
            df["anomaly_label"] = np.where(df["is_anomaly"], -1, 1)

            n_anomalies = df["is_anomaly"].sum()
            print(
                f"  🔴 Anomalies detected: {n_anomalies}/{len(df)} ({100 * n_anomalies / len(df):.1f}%)"
            )
            return df

        def fit_predict(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
            """
            Train on the healthy portion, then score the whole series.

            Parameters
            ----------
            df : pandas.DataFrame
                Full feature table, in time order.
            feature_cols : list
                Columns to train and score on.

            Returns
            -------
            pandas.DataFrame
                The table with the three result columns appended.
            """
            self.fit(df, feature_cols)
            return self.predict(df)


# ─── Pipeline: Select Features & Run Detection ─────────────────────────────


def select_features(df: pd.DataFrame, bearing_prefix: str = None) -> list:
    """
    Select the most informative feature columns.

    If bearing_prefix given (e.g. 'Bearing3'), focus on that bearing.
    Otherwise, use all RMS, kurtosis, and rolling features.
    """
    priority_keywords = ["_rms", "_kurt", "_std", "_max"]

    if bearing_prefix:
        cols = [
            c
            for c in df.columns
            if c.startswith(bearing_prefix) and any(k in c for k in priority_keywords)
        ]
        # Fallback: if prefix matches nothing, use all indicator columns
        if not cols:
            cols = [
                c
                for c in df.columns
                if any(k in c for k in priority_keywords)
                and "filename" not in c
                and "timestamp" not in c
            ]
    else:
        cols = [
            c
            for c in df.columns
            if any(k in c for k in priority_keywords)
            and "filename" not in c
            and "timestamp" not in c
        ]

    # Exclude result columns
    cols = [c for c in cols if c not in ["anomaly_score", "anomaly_label", "is_anomaly"]]
    return cols


FEATURE_SOURCES = ("auto", "enriched", "basic")


def resolve_feature_table(test_id: int, feature_source: str = "auto") -> tuple[pd.DataFrame, str]:
    """
    Load the feature table for one test and report which table was used.

    Two tables can back a detection run and they do not produce the same answer.
    ``data/processed/test{N}_raw.csv`` is committed and carries the six summary
    statistics per channel. ``data/processed/test{N}_features.csv`` is written by
    ``features.enrich_processed``, adds rolling and first-difference columns, and
    is gitignored -- so it is absent from a fresh clone until the pipeline has
    been run. For the failed bearing of test 3 that is 20 selected feature
    columns against 4.

    A different feature count fits a different Isolation Forest and yields
    different anomaly scores, so choosing whichever table happened to be on disk
    would make every downstream number depend on an untracked file. The choice is
    therefore named in the return value and printed, and ``enriched``/``basic``
    let a caller pin it rather than inherit it from the filesystem.

    Parameters
    ----------
    test_id : int
        1, 2 or 3.
    feature_source : {'auto', 'enriched', 'basic'}
        ``auto`` prefers the enriched table and falls back to the committed one,
        announcing which it took. ``enriched`` requires the enriched table and
        raises if it is missing. ``basic`` always reads the committed table.

    Returns
    -------
    tuple of (pandas.DataFrame, str)
        The loaded table and the source actually used, ``'enriched'`` or
        ``'basic'``.

    Raises
    ------
    ValueError
        If ``feature_source`` is not one of ``FEATURE_SOURCES``.
    FileNotFoundError
        If ``feature_source='enriched'`` and the enriched table is absent.
    """
    if feature_source not in FEATURE_SOURCES:
        raise ValueError(f"feature_source must be one of {FEATURE_SOURCES}, got {feature_source!r}")

    config = TEST_CONFIG[test_id]
    features_path = config["output_file"].parent / f"test{test_id}_features.csv"

    if feature_source == "enriched" and not features_path.exists():
        raise FileNotFoundError(
            f"Enriched feature table not found: {features_path}\n"
            f"Run: python -m nasa_bearing_anomaly.features --test {test_id}"
        )

    if feature_source != "basic" and features_path.exists():
        df = pd.read_csv(features_path, index_col="file_index")
        print(f"Feature source: enriched ({features_path.name}, {len(df.columns)} columns)")
        return df, "enriched"

    df = load_processed(test_id)
    print(f"Feature source: basic ({config['output_file'].name}, {len(df.columns)} columns)")
    if feature_source == "auto":
        print(
            "  Note: the enriched table is absent, so this run uses the committed "
            "summary table.\n"
            "  Rolling and first-difference features are not included and the "
            "anomaly scores will\n"
            "  differ from a run made against the enriched table. Run "
            f"'python -m nasa_bearing_anomaly.features --test {test_id}' first to "
            "match the published numbers."
        )
    return df, "basic"


def run_pipeline(
    test_id: int, method: str = "isolation_forest", feature_source: str = "auto"
) -> pd.DataFrame:
    """
    Full detection pipeline for a single test.

    Parameters
    ----------
    test_id : int
        1, 2, or 3.
    method : {'isolation_forest', 'autoencoder'}
        Detector to fit.
    feature_source : {'auto', 'enriched', 'basic'}
        Which feature table to read. See :func:`resolve_feature_table`.

    Returns
    -------
    pandas.DataFrame
        The input table with ``anomaly_score``, ``anomaly_label`` and
        ``is_anomaly`` appended.
    """
    config = TEST_CONFIG[test_id]

    df, source_used = resolve_feature_table(test_id, feature_source)

    # Select feature columns
    failed = config["failed_bearing"]
    feature_cols = select_features(df, bearing_prefix=failed)
    if len(feature_cols) < 2:
        feature_cols = select_features(df)  # fallback: all bearings

    print(
        f"Failed bearing: {failed} | Features selected: {len(feature_cols)} (source: {source_used})"
    )

    # Run detector
    contamination = 0.05 if test_id in [1, 2] else 0.08  # Test 3 degrades more gradually

    if method == "isolation_forest":
        detector = IsolationForestDetector(contamination=contamination)
        df_result = detector.fit_predict(df, feature_cols)
        # Save model
        save_dir = REPORTS_DIR / f"test{test_id}_model"
        detector.save(save_dir)

    elif method == "autoencoder":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for autoencoder. pip install -e '.[deep]'")
        detector = AutoencoderDetector(epochs=150)
        df_result = detector.fit_predict(df, feature_cols)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Save results
    out_path = REPORTS_DIR / f"test{test_id}_{method}_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(out_path)
    print(f"📊 Results saved: {out_path}")

    return df_result


# ─── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection on NASA Bearing Dataset")
    parser.add_argument("--test", type=str, default="all", help="1, 2, 3, or 'all'")
    parser.add_argument(
        "--method",
        type=str,
        default="isolation_forest",
        choices=["isolation_forest", "autoencoder"],
        help="Detection method",
    )
    parser.add_argument(
        "--feature-source",
        type=str,
        default="auto",
        choices=list(FEATURE_SOURCES),
        help=(
            "Which feature table to read: 'auto' prefers the enriched table, "
            "'enriched' requires it, 'basic' forces the committed summary table"
        ),
    )
    args = parser.parse_args()

    tests = [1, 2, 3] if args.test == "all" else [int(args.test)]

    for t in tests:
        print(f"\n{'=' * 60}")
        print(f"Running {args.method} on Test {t}...")
        print(f"{'=' * 60}")
        run_pipeline(t, method=args.method, feature_source=args.feature_source)
