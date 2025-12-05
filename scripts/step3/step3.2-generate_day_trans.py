# step3.2-generate_day_trans.py
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_in: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 4, dim_feedforward: int = 256):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model

        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_in)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        out = self.output_proj(h)
        return out


def main():
    base = Path(__file__).resolve().parents[2]
    proc = base / "processed"

    # Load original mapped time series for metadata and warm-up
    data = np.load(proc / "timeseries_mapped.npz", allow_pickle=True)
    t_train = data["t"]           # (T,)
    P_train = data["P"]           # (T, N)
    Q_train = data["Q"]           # (T, N)
    columns = data["columns"]     # (N,)

    T, N = P_train.shape
    D = 2 * N

    # Load normalization
    norm = np.load(proc / model / transformer / "ts_transformer_norm.npz", allow_pickle=True)
    x_mean = norm["x_mean"]       # (D,)
    x_std = norm["x_std"]         # (D,)
    context_len = int(norm["context_len"][0])

    # Build model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(d_in=D, d_model=128, nhead=4,
                                  num_layers=4, dim_feedforward=256)
    model_path = proc / "ts_transformer.pt"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Build normalized full state X_norm from training data
    X_train = np.concatenate([P_train, Q_train], axis=1)  # (T, D)
    X_norm = (X_train - x_mean) / x_std

    # Warm-up window: first context_len time steps from training
    if T < context_len:
        raise RuntimeError("Not enough data for warm-up window.")

    warmup = X_norm[:context_len, :].astype(np.float32)  # (L, D)

    # Generate synthetic day of M minutes
    M = 1440
    generated_norm = []

    # We will iteratively extend the sequence:
    # start_seq: last context_len steps
    current_seq = warmup.copy()  # (L, D)

    with torch.no_grad():
        for step in range(M):
            inp = torch.from_numpy(current_seq).unsqueeze(0).to(device)  # (1, L, D)
            pred = model(inp)  # (1, L, D)
            # next step prediction is last time step in the window
            next_norm = pred[0, -1, :].cpu().numpy()  # (D,)
            generated_norm.append(next_norm)

            # slide the window
            current_seq = np.vstack([current_seq[1:, :], next_norm[None, :]])

    generated_norm = np.stack(generated_norm, axis=0)  # (M, D)

    # De-normalize
    X_syn = generated_norm * x_std + x_mean           # (M, D)

    # Split back to P, Q
    P_syn = X_syn[:, :N]
    Q_syn = X_syn[:, N:]

    # Enforce simple sign constraints using meta
    meta = pd.read_csv(proc / "timeseries_meta.csv")
    assert len(meta) == N

    for j, row in meta.iterrows():
        table = row["table"]
        if table == "load":
            P_syn[:, j] = np.maximum(P_syn[:, j], 0.0)
        elif table in ("gen", "sgen"):
            P_syn[:, j] = np.maximum(P_syn[:, j], 0.0)
        # shunts left as-is (P≈0, Q meaningful)

    # Build timestamps for synthetic day: directly after training period
    t_series = pd.to_datetime(t_train)
    start_time = t_series.max() + pd.Timedelta(minutes=1)
    t_syn = pd.date_range(start=start_time, periods=M, freq="1min")

    # Save in same format as original synthetic_day_001.npz
    out_path = proc / synthetic / "day001_trans.npz"
    np.savez(
        out_path,
        t=t_syn.to_numpy("datetime64[ns]"),
        P=P_syn,
        Q=Q_syn,
        columns=columns,
    )

    print("Synthetic day (LLM) saved to:", out_path)
    print("Shapes: P_syn", P_syn.shape, "Q_syn", Q_syn.shape)
    print("Columns (first 10):", list(columns[:10]))


if __name__ == "__main__":
    main()
