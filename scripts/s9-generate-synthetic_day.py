#s9-generate-synthetic_day.py
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    base = Path(__file__).resolve().parents[1]
    proc = base / "processed"

    # Load mapped time series (for metadata and column order)
    ts = np.load(proc / "timeseries_mapped.npz", allow_pickle=True)
    t_train = ts["t"]              # (T,)
    columns = ts["columns"]        # (N,)
    P_train = ts["P"]              # (T, N)
    Q_train = ts["Q"]              # (T, N)

    T, N = P_train.shape
    assert Q_train.shape == (T, N)
    D = 2 * N

    # Load statistical model
    model = np.load(proc / "stat_model.npz", allow_pickle=True)
    mu = model["mu"]               # (1440, D)
    r_mean = model["r_mean"]       # (D,)
    r_std = model["r_std"]         # (D,)
    pca_components = model["pca_components"]  # (K, D)
    pca_mean = model["pca_mean"]              # (D,)
    A = model["A"]                 # (K, K)
    cov_eps = model["cov_eps"]     # (K, K)

    K = pca_components.shape[0]
    assert pca_components.shape[1] == D

    # Length of synthetic day
    M = 1440  # minutes

    # Latent VAR(1) simulation: z_t in R^K
    rng = np.random.default_rng(0)
    Z_syn = np.zeros((M, K), dtype=float)

    # Initialize z_0 from epsilon distribution
    eps0 = rng.multivariate_normal(mean=np.zeros(K), cov=cov_eps)
    Z_syn[0, :] = eps0

    for t in range(1, M):
        eps_t = rng.multivariate_normal(mean=np.zeros(K), cov=cov_eps)
        Z_syn[t, :] = Z_syn[t - 1, :] @ A + eps_t

    # Reconstruct standardized residuals R_std from PCA:
    # R_std ≈ Z @ components + pca_mean
    R_std_syn = Z_syn @ pca_components + pca_mean  # (M, D)

    # De-standardize residuals
    R_syn = R_std_syn * r_std + r_mean            # (M, D)

    # Add daily profile mu[minute, :]
    # minutes 0..1439 correspond directly
    X_syn = mu[:M, :] + R_syn                     # (M, D)

    # Split back to P, Q
    P_syn = X_syn[:, :N]
    Q_syn = X_syn[:, N:]

    # Enforce simple sign constraints using meta info
    meta = pd.read_csv(proc / "timeseries_meta.csv")
    assert len(meta) == N

    for j, row in meta.iterrows():
        table = row["table"]
        # Loads: P >= 0
        if table == "load":
            P_syn[:, j] = np.maximum(P_syn[:, j], 0.0)
        # Generators / sgens: P >= 0 (no pumping in this simple model)
        elif table in ("gen", "sgen"):
            P_syn[:, j] = np.maximum(P_syn[:, j], 0.0)
        # Shunts: leave P (should be ~0 anyway), only Q is meaningful

    # Build timestamps for synthetic day: directly after training period
    t_series = pd.to_datetime(t_train)
    start_time = t_series.max() + pd.Timedelta(minutes=1)
    t_syn = pd.date_range(start=start_time, periods=M, freq="1min")

    # Save
    out_path = proc / "synthetic_day_001.npz"
    np.savez(
        out_path,
        t=t_syn.to_numpy("datetime64[ns]"),
        P=P_syn,
        Q=Q_syn,
        columns=columns,
    )

    # Quick text summary
    print("Synthetic day saved to:", out_path)
    print("Shapes: P_syn", P_syn.shape, "Q_syn", Q_syn.shape)
    print("Columns (first 10):", list(columns[:10]))


if __name__ == "__main__":
    main()
