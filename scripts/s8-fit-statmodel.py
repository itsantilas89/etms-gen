#s8-fit-statmodel.py
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def main():
    base = Path(__file__).resolve().parents[1]
    proc = base / "processed"

    data = np.load(proc / "timeseries_mapped.npz", allow_pickle=True)
    t = data["t"]              # datetime64[ns], shape (T,)
    P = data["P"]              # shape (T, N)
    Q = data["Q"]              # shape (T, N)
    columns = data["columns"]  # shape (N,)

    T, N = P.shape
    assert Q.shape == (T, N)

    # 1) Build full state x_t = [P_t, Q_t]
    X = np.concatenate([P, Q], axis=1)   # shape (T, 2N)
    D = X.shape[1]

    # 2) Minute-of-day indices (0..1439)
    # t_series is a DatetimeIndex, so use .hour / .minute directly
    t_series = pd.to_datetime(t)
    minute_of_day = t_series.hour * 60 + t_series.minute
    minute_of_day = minute_of_day.to_numpy()  # shape (T,)

    if T != 2880:
        print(f"Warning: expected 2880 samples, got {T}")

    # 3) Compute typical daily profile mu[minute, dim]
    mu = np.zeros((1440, D), dtype=float)
    for minute in range(1440):
        idx = np.where(minute_of_day == minute)[0]
        if len(idx) == 0:
            # should not happen with exactly 2 days; fallback to zeros
            mu[minute, :] = 0.0
        else:
            mu[minute, :] = X[idx, :].mean(axis=0)

    # 4) Compute residuals R[t, :] = X[t, :] - mu[minute_of_day[t], :]
    R = np.empty_like(X)
    for i in range(T):
        R[i, :] = X[i, :] - mu[minute_of_day[i], :]

    # 5) Standardize residuals per dimension
    r_mean = R.mean(axis=0)
    r_std = R.std(axis=0)
    r_std[r_std == 0.0] = 1.0  # avoid div-by-zero

    R_std = (R - r_mean) / r_std

    # 6) PCA on standardized residuals -> low-dimensional latent z_t
    K = min(10, D, T - 1)  # max 10 components, and <= D, <= T-1
    pca = PCA(n_components=K)
    Z = pca.fit_transform(R_std)   # shape (T, K)

    # 7) Fit VAR(1): Z_next = Z_prev @ A
    Z_prev = Z[:-1, :]   # shape (T-1, K)
    Z_next = Z[1:, :]    # shape (T-1, K)

    A, _, _, _ = np.linalg.lstsq(Z_prev, Z_next, rcond=None)  # A: (K, K)

    # residuals in latent space
    eps = Z_next - Z_prev @ A
    cov_eps = np.cov(eps, rowvar=False)  # shape (K, K)

    # 8) Save everything
    out = proc / "stat_model.npz"
    np.savez(
        out,
        columns=columns,
        mu=mu,
        r_mean=r_mean,
        r_std=r_std,
        pca_components=pca.components_,  # shape (K, D)
        pca_mean=pca.mean_,              # shape (D,)
        A=A,
        cov_eps=cov_eps,
    )

    print("Saved statistical model to:", out)
    print("State dims: T =", T, "N =", N, "D =", D, "K =", K)


if __name__ == "__main__":
    main()
