import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from .registry import register_model


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


@register_model('BiLSTM')
class LSTMClassModel(BaseModel):
    def __init__(self, seq_len=10, hidden=16, epochs=5, lr=5e-4, device='cpu'):
        super().__init__('BiLSTM')
        self.seq_len = seq_len  # Réduit de 20 à 10
        self.hidden = hidden    # Réduit de 24 à 16
        self.epochs = epochs    # Réduit de 10 à 5
        self.lr = lr            # Réduit de 1e-3 à 5e-4
        self.device = torch.device(device)
        self.net = None
        self.sc = StandardScaler()
        self._buffer = []

    def fit(self, X, y):
        if len(X) <= self.seq_len + 10 or len(np.unique(y)) < 2:
            return
        Xs = self.sc.fit_transform(X).astype(np.float32)
        n = len(Xs) - self.seq_len
        if n <= 0:
            return
        X_seq = np.stack([Xs[i:i + self.seq_len] for i in range(n)])
        y_seq = y[self.seq_len:].astype(np.float32)
        self.net = _BiLSTMNet(X.shape[1], self.hidden).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-3)  # Augmenté de 1e-4 à 1e-3
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
        if not self.is_fitted or self.net is None:
            return 0.5, 0.25
        x = np.array(X_row).flatten()
        self._buffer.append(x)
        if len(self._buffer) < self.seq_len:
            return 0.5, 0.25
        seq = np.array(self._buffer[-self.seq_len:])
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


@register_model('GraphNN')
class GraphNNModel(BaseModel):
    def __init__(self, epochs=50, lr=1e-3, hidden=32, device='cpu'):
        super().__init__('GraphNN')
        self.epochs = epochs
        self.lr = lr
        self.hidden = hidden
        self.device = torch.device(device)
        self.net = None
        self.sc = StandardScaler()
        self.edge_index = None
        self._available = True
        try:
            import torch_geometric
        except ImportError:
            self._available = False

    def _build_edge_index(self, num_nodes):
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
