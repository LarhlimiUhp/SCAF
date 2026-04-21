import numpy as np
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from .registry import register_model


if _TORCH_AVAILABLE:
    class _BiLSTMNet(nn.Module):
        def __init__(self, input_dim, hidden=24, n_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                                dropout=dropout if n_layers > 1 else 0.0,
                                bidirectional=True)
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, 12), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(12, 1), nn.Sigmoid())

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)
else:
    _BiLSTMNet = None  # type: ignore[assignment,misc]


@register_model('BiLSTM')
class LSTMClassModel(BaseModel):
    def __init__(self, seq_len=10, hidden=16, epochs=5, lr=5e-4, device='cpu'):
        super().__init__('BiLSTM')
        self.seq_len = seq_len
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device(device) if _TORCH_AVAILABLE else None
        self.net = None
        self.sc = StandardScaler()
        self._buffer = []
        # sklearn fallback used when PyTorch is unavailable
        self._fallback_mlp = None

    def fit(self, X, y):
        if len(X) <= self.seq_len + 10 or len(np.unique(y)) < 2:
            return

        if not _TORCH_AVAILABLE or _BiLSTMNet is None:
            # Sklearn fallback: MLP on flattened sliding-window (lag) features.
            # This approximates the temporal awareness of BiLSTM by presenting
            # the last seq_len timesteps as a concatenated feature vector.
            from sklearn.neural_network import MLPClassifier
            Xs = self.sc.fit_transform(X).astype(np.float32)
            n = len(Xs) - self.seq_len
            if n <= 10:
                return
            X_seq = np.stack([Xs[i:i + self.seq_len].flatten() for i in range(n)])
            y_seq = y[self.seq_len:]
            if len(np.unique(y_seq)) < 2:
                return
            self._fallback_mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
            )
            try:
                self._fallback_mlp.fit(X_seq, y_seq)
                self.is_fitted = True
            except Exception:
                self.is_fitted = False
            return

        Xs = self.sc.fit_transform(X).astype(np.float32)
        n = len(Xs) - self.seq_len
        if n <= 0:
            return
        X_seq = np.stack([Xs[i:i + self.seq_len] for i in range(n)])
        y_seq = y[self.seq_len:].astype(np.float32)
        self.net = _BiLSTMNet(X.shape[1], self.hidden).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        crit = nn.BCELoss()
        self.net.train()
        for _ in range(self.epochs):
            tensor_x = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            tensor_y = torch.tensor(y_seq, dtype=torch.float32).to(self.device)
            opt.zero_grad()
            loss = crit(self.net(tensor_x), tensor_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
        self.net.eval()
        self.is_fitted = True

    def predict_proba_one(self, X_row):
        if not self.is_fitted:
            return 0.5, 0.25
        x = np.array(X_row).flatten()
        self._buffer.append(x)
        if len(self._buffer) < self.seq_len:
            return 0.5, 0.25
        seq = np.array(self._buffer[-self.seq_len:])

        if self._fallback_mlp is not None:
            # sklearn fallback inference
            try:
                xs = self.sc.transform(seq).astype(np.float32).flatten()
                p = float(self._fallback_mlp.predict_proba(xs.reshape(1, -1))[0, 1])
                return p, p * (1 - p)
            except Exception:
                return 0.5, 0.25

        try:
            xs = self.sc.transform(seq).astype(np.float32)
            t = torch.tensor(xs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p = float(self.net(t).item())
            return p, p * (1 - p)
        except Exception:
            return 0.5, 0.25


@register_model('TabNet')
class TabNetModel(BaseModel):
    def __init__(self, epochs=20, lr=2e-2, device='cpu'):
        super().__init__('TabNet')
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.model = None
        self._raw = None

        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            self._raw = TabNetClassifier(optimizer_params={
                'lr': self.lr,
            }, verbose=0)
        except ImportError:
            self._raw = None

    def fit(self, X, y):
        if self._raw is None or len(X) < 100 or len(np.unique(y)) < 2:
            return
        try:
            self._raw.fit(X, y, max_epochs=self.epochs, patience=10, batch_size=64)
            self.model = self._raw
            self.is_fitted = True
        except Exception:
            self.is_fitted = False

    def predict_proba_one(self, X_row):
        if not self.is_fitted or self.model is None:
            return 0.5, 0.25
        try:
            p = float(self.model.predict_proba(X_row)[0, 1])
            return p, p * (1 - p)
        except Exception:
            return 0.5, 0.25


if _TORCH_AVAILABLE:
    class _GraphConvNet(nn.Module):
        def __init__(self, input_dim, hidden=32, n_layers=2):
            super().__init__()
            try:
                from torch_geometric.nn import GCNConv, global_mean_pool
            except ImportError:
                raise ImportError('torch_geometric is required for GraphNNModel')

            self.convs = nn.ModuleList()
            for i in range(n_layers):
                self.convs.append(
                    GCNConv(input_dim if i == 0 else hidden, hidden)
                )
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1), nn.Sigmoid())
            self.global_mean_pool = global_mean_pool

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for conv in self.convs:
                x = conv(x, edge_index)
                x = torch.relu(x)
            x = self.global_mean_pool(x, batch)
            return self.head(x).squeeze(-1)
else:
    _GraphConvNet = None  # type: ignore[assignment,misc]


@register_model('GraphNN')
class GraphNNModel(BaseModel):
    def __init__(self, epochs=50, lr=1e-3, hidden=32, device='cpu'):
        super().__init__('GraphNN')
        self.epochs = epochs
        self.lr = lr
        self.hidden = hidden
        self.device = torch.device(device) if _TORCH_AVAILABLE else None
        self.net = None
        self.sc = StandardScaler()
        self.edge_index = None
        self._available = _TORCH_AVAILABLE
        if self._available:
            try:
                import torch_geometric  # noqa: F401
            except ImportError:
                self._available = False

    def _build_edge_index(self, num_nodes):
        if not _TORCH_AVAILABLE:
            return None
        row = torch.arange(num_nodes).repeat_interleave(num_nodes)
        col = torch.arange(num_nodes).repeat(num_nodes)
        return torch.stack([row, col], dim=0)

    def fit(self, X, y):
        if not self._available or len(X) < 100 or len(np.unique(y)) < 2:
            return
        try:
            from torch_geometric.data import Data, DataLoader
            Xs = self.sc.fit_transform(X).astype(np.float32)
            num_nodes = Xs.shape[1]
            self.edge_index = self._build_edge_index(num_nodes).to(self.device)
            dataset = []
            for i in range(len(Xs)):
                x = torch.tensor(Xs[i].reshape(-1, 1), dtype=torch.float32)
                y_t = torch.tensor([y[i]], dtype=torch.float32)
                dataset.append(Data(x=x, edge_index=self.edge_index, y=y_t))
            self.net = _GraphConvNet(num_nodes, hidden=self.hidden).to(self.device)
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
            loss_fn = nn.BCELoss()
            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            self.net.train()
            for _ in range(self.epochs):
                for batch in loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    out = self.net(batch)
                    loss = loss_fn(out, batch.y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    optimizer.step()
            self.net.eval()
            self.is_fitted = True
        except Exception:
            self.is_fitted = False

    def predict_proba_one(self, X_row):
        if not self.is_fitted or self.net is None:
            return 0.5, 0.25
        try:
            from torch_geometric.data import Data
            x = np.array(X_row).flatten().astype(np.float32)
            x = self.sc.transform(x.reshape(1, -1)).reshape(-1, 1)
            data = Data(x=torch.tensor(x), edge_index=self.edge_index)
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            data = data.to(self.device)
            with torch.no_grad():
                p = float(self.net(data).item())
            return p, p * (1 - p)
        except Exception:
            return 0.5, 0.25
