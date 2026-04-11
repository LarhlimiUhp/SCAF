"""
Deep Learning-based Regime Detector for SCAF-LS

Uses a Bidirectional LSTM classifier to map a sliding window of market
features onto one of four regimes: bull, bear, sideways, crisis.

Training labels are derived from the rule-based heuristic
(`_detect_regime`) applied to the *training* window, so the model learns
a smooth, high-capacity approximation of regime transitions directly from
raw features — without requiring manually labelled data.

Falls back gracefully when PyTorch is not available.
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

# ── rule-based labeller (same logic as pipeline helper) ─────────────────── #

def _heuristic_regime(vol_5d: float, vix: float = 20.0) -> str:
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
        is a volatility-like feature. If *vix_col* is given that column is
        used as VIX.
    """
    labels = np.empty(len(X), dtype=np.int64)
    for i, row in enumerate(X):
        vol = abs(float(row[vol_col]))
        vix = float(row[vix_col]) if vix_col is not None and vix_col < len(row) else 20.0
        regime = _heuristic_regime(vol, vix)
        labels[i] = _LABEL_TO_IDX[regime]
    return labels


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
    """DL-based regime detector wrapping a BiLSTM classifier.

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
        Index of the volatility column in the feature matrix (used for
        generating training labels via heuristic).
    vix_col:
        Index of the VIX column in the feature matrix (optional).
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

    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "RegimeDetector":
        """Train the BiLSTM on *X* using heuristic-derived labels.

        Parameters
        ----------
        X:
            Feature matrix (n_samples, n_features).
        """
        if not _TORCH_AVAILABLE:
            logger.warning("RegimeDetector: PyTorch not available — falling back to heuristic.")
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

        Returns
        -------
        One of "bull", "bear", "sideways", "crisis".
        """
        if not self._fitted or self._net is None:
            return self._heuristic_fallback(X_window)

        try:
            window = X_window[-self.seq_len:] if len(X_window) >= self.seq_len else X_window
            if len(window) < self.seq_len:
                return self._heuristic_fallback(X_window)

            X_norm = (window - self._mean) / self._std  # type: ignore[operator]
            x = torch.tensor(X_norm[np.newaxis].astype(np.float32))
            with torch.no_grad():
                logits = self._net(x)
                idx = int(torch.argmax(logits, dim=1).item())
            return _IDX_TO_LABEL[idx]
        except Exception as exc:
            logger.warning("RegimeDetector.predict failed: %s", exc)
            return self._heuristic_fallback(X_window)

    # ------------------------------------------------------------------ #

    def _heuristic_fallback(self, X_window: np.ndarray) -> str:
        """Return a heuristic regime when DL is unavailable."""
        if len(X_window) == 0:
            return "sideways"
        row = X_window[-1]
        vol = abs(float(row[self.vol_col])) if len(row) > self.vol_col else 0.01
        vix = float(row[self.vix_col]) if self.vix_col is not None and len(row) > self.vix_col else 20.0
        return _heuristic_regime(vol, vix)
