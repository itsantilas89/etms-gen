from pathlib import Path
import json

import numpy as np
import pandas as pd


def main():
    base = Path(__file__).resolve().parents[1]
    proc = base / "processed"

    # Load aligned P, Q and mapping
    p_all = pd.read_parquet(proc / "P_all.parquet")
    q_all = pd.read_parquet(proc / "Q_all.parquet")

    with (proc / "mapping.json").open("r", encoding="utf-8") as f:
        mapping = json.load(f)  # col_name -> [table, idx]

    # Use only columns that are actually mapped
    mapped_cols = sorted(mapping.keys())

    print("Timestamps:", p_all.shape[0])
    print("Mapped equipment columns:", len(mapped_cols))

    # Sanity: ensure Q has the same mapped columns
    missing_in_q = [c for c in mapped_cols if c not in q_all.columns]
    if missing_in_q:
        raise RuntimeError(f"Columns mapped but missing in Q_all: {missing_in_q}")

    # Extract timestamps (already aligned in s5)
    t = p_all["TIMESTAMP"].to_numpy()

    # Build P and Q matrices: shape (T, N_equipment)
    P = p_all[mapped_cols].to_numpy(dtype=float)
    Q = q_all[mapped_cols].to_numpy(dtype=float)

    print("P shape:", P.shape)  # (T, N)
    print("Q shape:", Q.shape)

    # Optionally, also store a compact metadata table
    meta_rows = []
    for col in mapped_cols:
        tbl, idx = mapping[col]
        meta_rows.append({
            "column": col,
            "table": tbl,
            "pp_index": idx,
        })
    meta = pd.DataFrame(meta_rows)

    # Save to npz + csv for later use
    out_npz = proc / "timeseries_mapped.npz"
    np.savez_compressed(out_npz, t=t, P=P, Q=Q, columns=np.array(mapped_cols))

    meta_path = proc / "timeseries_meta.csv"
    meta.to_csv(meta_path, index=False)

    print("Saved:")
    print("  ", out_npz)
    print("  ", meta_path)


if __name__ == "__main__":
    main()
