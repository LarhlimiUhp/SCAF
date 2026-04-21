"""
Deep Learning-based Regime Detector for SCAF-LS

Uses a Bidirectional LSTM classifier to map a sliding window of market
features onto one of four regimes: bull, bear, sideways, crisis.

Training labels are derived from the rule-based heuristic
(`_detect_regime`) applied to the *training* window, so the model learns
a smooth, high-capacity approximation of regime transitions directly from
raw features — without requiring manually labelled data.

When PyTorch is not available the detector falls back to a
GaussianMixture (GMM) regime classifier that is fit on the training data
and maps the 4 components to regime labels by ascending volatility
centroid. This avoids applying absolute-scale thresholds to z-scored
feature columns, which caused the heuristic fallback to mis-classify most
regimes as "bull".
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── optional torch ──────────────────────────────────────────────────────── #
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

REGIME_LABELS = ["bull", "bear", "sideways", "crisis"]
_LABEL_TO_IDX = {r: i for i, r in enumerate(REGIME_LABELS)}
_IDX_TO_LABEL = {i: r for r, i in _LABEL_TO_IDX.items()}

# ── rule-based labeller (operates on RAW / un-normalised features) ────────── #

def _heuristic_regime(vol_5d: float, vix: float = 20.0) -> str:
    """Classify regime from *raw* (un-normalised) vol and VIX values."""
    if vix > 35 or vol_5d > 0.03:
        return "crisis"
    if vol_5d > 0.015:
        return "bear"
    if vol_5d < 0.007:
        return "bull"
    return "sideways"


def _label_sequence(X: np.ndarray, vol_col: int = 0, vix_col: Optional[int] = None) -> np.ndarray:
    """Generate per-step regime labels from feature matrix rows.

    Parameters
    ----------
    X:
        Feature matrix (n_samples, n_features).  Assumes the first column
        is a volatility-like feature in its *raw* (un-normalised) scale.
        If *vix_col* is given that column is used as VIX (also raw).
    """
    labels = np.empty(len(X), dtype=np.int64)
    for i, row in enumerate(X):
        vol = abs(float(row[vol_col]))
        vix = float(row[vix_col]) if vix_col is not None and vix_col < len(row) else 20.0
        regime = _heuristic_regime(vol, vix)
        labels[i] = _LABEL_TO_IDX[regime]
    return labels


# ── GMM-based sklearn fallback ───────────────────────────────────────────── #

def _build_gmm_detector(
    X: np.ndarray,
    vol_col: int = 0,
    vix_col: Optional[int] = None,
):
    """Fit a GaussianMixture (4 components) on (vol, vix) features and
    return (gmm, component_to_label, mean_, std_) tuple.

    Components are sorted by ascending mean volatility and mapped to
    ["bull", "sideways", "bear", "crisis"] respectively.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    # Select columns: vol + vix (or just vol if vix missing)
    cols = [vol_col]
    if vix_col is not None and vix_col < X.shape[1] and vix_col != vol_col:
        cols.append(vix_col)
    Xg = X[:, cols].astype(np.float64)

    scaler = StandardScaler()
    Xg_scaled = scaler.fit_transform(Xg)

    gmm = GaussianMixture(n_components=4, covariance_type="full",
                          random_state=42, n_init=3, max_iter=200)
    gmm.fit(Xg_scaled)

    # Sort components by mean of vol dimension (ascending → bull … crisis)
    vol_means = gmm.means_[:, 0]  # vol is always first selected col
    order = np.argsort(vol_means)       # ascending
    component_to_label = {int(comp): lbl
                          for comp, lbl in zip(order, ["bull", "sideways", "bear", "crisis"])}

    return gmm, component_to_label, scaler, cols


# ── PyTorch architecture ─────────────────────────────────────────────────── #

if _TORCH_AVAILABLE:
    class _RegimeBiLSTM(nn.Module):
        """Bidirectional LSTM → 4-class regime classifier."""

        def __init__(self, input_dim: int, hidden: int = 32, n_layers: int = 2,
                     n_classes: int = 4, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden, n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, n_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])   # last timestep logits
else:
    _RegimeBiLSTM = None  # type: ignore[assignment,misc]


# ── RegimeDetector ──────────────────────────────────────────────────────── #

class RegimeDetector:
    """Regime detector: BiLSTM (primary) with GMM sklearn fallback.

    When PyTorch is available a bidirectional LSTM is trained using
    heuristic-derived labels.  When PyTorch is unavailable a
    GaussianMixture (4 components) is fitted on the raw feature matrix
    and components are mapped to regimes by ascending volatility centroid.
    This avoids the previous bug of applying absolute-scale thresholds to
    z-scored features.

    Parameters
    ----------
    seq_len:
        Number of time-steps fed to the LSTM at each inference call.
    hidden:
        Hidden size of the BiLSTM.
    n_layers:
        Number of stacked BiLSTM layers.
    epochs:
        Training epochs.
    lr:
        Learning rate for Adam.
    vol_col:
        Index of the volatility column in the feature matrix (raw scale,
        used for heuristic label generation and GMM fitting).
    vix_col:
        Index of the VIX column in the feature matrix (optional, raw scale).
    """

    def __init__(
        self,
        seq_len: int = 10,
        hidden: int = 32,
        n_layers: int = 2,
        epochs: int = 10,
        lr: float = 1e-3,
        vol_col: int = 0,
        vix_col: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.vol_col = vol_col
        self.vix_col = vix_col
        self._net = None
        self._fitted = False
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        # GMM fallback attributes
        self._gmm = None
        self._gmm_component_to_label: Optional[dict] = None
        self._gmm_scaler = None
        self._gmm_cols: Optional[list] = None
        self._gmm_fitted = False

    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "RegimeDetector":
        """Train the regime detector on *X* (raw / un-normalised features).

        Trains BiLSTM when PyTorch is available; otherwise fits a GMM
        on the volatility and VIX columns.

        Parameters
        ----------
        X:
            Feature matrix (n_samples, n_features) in **raw** (un-scaled)
            units so that heuristic thresholds and GMM centroids are
            interpretable.
        """
        if not _TORCH_AVAILABLE:
            # ── GMM fallback ─────────────────────────────────────────── #
            try:
                (self._gmm, self._gmm_component_to_label,
                 self._gmm_scaler, self._gmm_cols) = _build_gmm_detector(
                    X, self.vol_col, self.vix_col
                )
                self._gmm_fitted = True
                logger.info(
                    "RegimeDetector: PyTorch unavailable — GMM fallback fitted "
                    "(cols=%s, label_map=%s).",
                    self._gmm_cols, self._gmm_component_to_label,
                )
            except Exception as exc:
                logger.warning("RegimeDetector GMM fallback failed: %s", exc)
                self._gmm_fitted = False
            self._fitted = False
            return self

        n, d = X.shape
        if n <= self.seq_len + 10:
            logger.warning("RegimeDetector: insufficient training samples (%d).", n)
            self._fitted = False
            return self

        # Z-score normalise (fit on training data only)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std

        # Build sequences + labels
        seqs, targets = [], []
        labels = _label_sequence(X, self.vol_col, self.vix_col)
        for i in range(self.seq_len, n):
            seqs.append(X_norm[i - self.seq_len: i])
            targets.append(labels[i])

        X_t = torch.tensor(np.array(seqs, dtype=np.float32))
        y_t = torch.tensor(np.array(targets, dtype=np.int64))

        # Model
        self._net = _RegimeBiLSTM(input_dim=d, hidden=self.hidden,
                                   n_layers=self.n_layers)
        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self._net.train()
        batch = 64
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X_t))
            total_loss = 0.0
            for start in range(0, len(X_t), batch):
                idx = perm[start: start + batch]
                xb, yb = X_t[idx], y_t[idx]
                optimiser.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
            logger.debug("RegimeDetector epoch %d/%d loss=%.4f", epoch + 1, self.epochs,
                         total_loss / max(1, len(X_t) // batch))

        self._net.eval()
        self._fitted = True
        logger.info("RegimeDetector trained on %d sequences (%d features).", len(seqs), d)
        return self

    def predict(self, X_window: np.ndarray) -> str:
        """Predict the current regime from a feature window.

        Parameters
        ----------
        X_window:
            Array of shape (seq_len, n_features) or (n_samples, n_features).
            If more than *seq_len* rows are provided the last *seq_len* rows
            are used.
            **Important**: pass the *raw* (un-scaled) feature matrix so the
            GMM fallback can compare against its fitted centroids.

        Returns
        -------
        One of "bull", "bear", "sideways", "crisis".
        """
        # ── (1) BiLSTM (primary, requires torch) ─────────────────────── #
        if self._fitted and self._net is not None:
            try:
                window = X_window[-self.seq_len:] if len(X_window) >= self.seq_len else X_window
                if len(window) < self.seq_len:
                    return self._fallback(X_window)

                X_norm = (window - self._mean) / self._std  # type: ignore[operator]
                x = torch.tensor(X_norm[np.newaxis].astype(np.float32))
                with torch.no_grad():
                    logits = self._net(x)
                    idx = int(torch.argmax(logits, dim=1).item())
                return _IDX_TO_LABEL[idx]
            except Exception as exc:
                logger.warning("RegimeDetector BiLSTM predict failed: %s", exc)

        # ── (2) GMM fallback (sklearn, no torch needed) ───────────────── #
        return self._fallback(X_window)

    # ------------------------------------------------------------------ #

    def _fallback(self, X_window: np.ndarray) -> str:
        """GMM-based fallback, then heuristic if GMM is not fitted."""
        if self._gmm_fitted and self._gmm is not None:
            try:
                # Use last row for point prediction
                row = X_window[-1:] if len(X_window) > 0 else X_window
                Xg = row[:, self._gmm_cols].astype(np.float64)  # type: ignore[index]
                Xg_scaled = self._gmm_scaler.transform(Xg)
                comp = int(self._gmm.predict(Xg_scaled)[0])
                return self._gmm_component_to_label.get(comp, "sideways")  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning("RegimeDetector GMM predict failed: %s", exc)

        # ── (3) Absolute-threshold heuristic ─────────────────────────── #
        return self._heuristic_fallback(X_window)

    def _heuristic_fallback(self, X_window: np.ndarray) -> str:
        """Return a heuristic regime when DL and GMM are unavailable.

        NOTE: works correctly only when X_window contains *raw* (un-scaled)
        feature values so that the absolute thresholds are meaningful.
        """
        if len(X_window) == 0:
            return "sideways"
        row = X_window[-1]
        vol = abs(float(row[self.vol_col])) if len(row) > self.vol_col else 0.01
        vix = float(row[self.vix_col]) if self.vix_col is not None and len(row) > self.vix_col else 20.0
        return _heuristic_regime(vol, vix)
