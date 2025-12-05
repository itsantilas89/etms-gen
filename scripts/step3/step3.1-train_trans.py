# step3.1-train_trans.py
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# paths
ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "processed"
MODELS_DIR = PROC_DIR / "models"
TRANSFORMER_DIR = MODELS_DIR / "transformer"

# -----------------------------------------------------------------------------


class TimeSeriesDataset(Dataset):
    def __init__(self, X_norm: np.ndarray, context_len: int):
        """
        X_norm: (T, D) normalized time series
        context_len: window length (L)
        We build samples: seq[0:L] and learn to predict seq[1:L] from seq[0:L-1].
        """
        self.X = X_norm.astype(np.float32)
        self.L = context_len
        self.T, self.D = self.X.shape
        if self.T <= self.L:
            raise ValueError("T must be > context_len")
        self.n_samples = self.T - self.L

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq = self.X[idx:idx + self.L, :]  # (L, D)
        return seq


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, L, d_model)
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_in: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 4,
                 dim_feedforward: int = 256):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model

        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_in)

    def forward(self, x):
        """
        x: (B, L, D_in)
        returns: (B, L, D_in)
        """
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        out = self.output_proj(h)
        return out


def main():
    # load mapped time series
    data = np.load(PROC_DIR / "timeseries_mapped.npz", allow_pickle=True)
    P = data["P"]              # (T, N)
    Q = data["Q"]              # (T, N)
    columns = data["columns"]  # (N,)

    T, N = P.shape
    assert Q.shape == (T, N)
    D = 2 * N

    # build full state X[t] = [P_t, Q_t]
    X = np.concatenate([P, Q], axis=1)  # (T, D)

    # normalize per dimension
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    X_norm = (X - x_mean) / x_std

    # dataset / loader
    context_len = 60
    batch_size = 64

    ds = TimeSeriesDataset(X_norm, context_len=context_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(d_in=D, d_model=128, nhead=4,
                                  num_layers=4, dim_feedforward=256)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_epochs = 50
    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for batch in dl:
            batch = batch.to(device)  # (B, L, D)
            pred = model(batch)       # (B, L, D)

            # next-step prediction within window
            pred_next = pred[:, :-1, :]
            target_next = batch[:, 1:, :]

            loss = loss_fn(pred_next, target_next)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch:3d} | train MSE: {avg_loss:.6f}")

    # ensure dirs exist
    TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)

    # save model + normalization
    model_path = TRANSFORMER_DIR / "ts_transformer.pt"
    torch.save(model.state_dict(), model_path)

    norm_path = TRANSFORMER_DIR / "ts_transformer_norm.npz"
    np.savez(
        norm_path,
        x_mean=x_mean,
        x_std=x_std,
        context_len=np.array([context_len], dtype=np.int32),
        columns=columns,
    )

    print("Saved Transformer model to:", model_path)
    print("Saved normalization to   :", norm_path)
    print("T =", T, "N =", N, "D =", D)


if __name__ == "__main__":
    main()
