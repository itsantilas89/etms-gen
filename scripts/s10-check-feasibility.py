#s10-check-feasibility.py
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp


def apply_snapshot_to_net(net, meta: pd.DataFrame, P_t: np.ndarray, Q_t: np.ndarray):
    """
    Write one time-step (P_t, Q_t) into the pandapower net according to meta.
    P_t, Q_t are 1D arrays of length N (same column order as meta).
    """
    for j, row in meta.iterrows():
        table = row["table"]
        idx = int(row["pp_index"])
        p = float(P_t[j])
        q = float(Q_t[j])

        if table == "load":
            net.load.at[idx, "p_mw"] = max(p, 0.0)  # load consumption >= 0
            net.load.at[idx, "q_mvar"] = q
        elif table == "sgen":
            # RES / static generators: positive P means generation
            net.sgen.at[idx, "p_mw"] = max(p, 0.0)
            net.sgen.at[idx, "q_mvar"] = q
        elif table == "gen":
            # Conventional generators
            net.gen.at[idx, "p_mw"] = max(p, 0.0)
            net.gen.at[idx, "q_mvar"] = q
        elif table == "shunt":
            # Shunts: only Q is relevant, set P to zero
            net.shunt.at[idx, "p_mw"] = 0.0
            net.shunt.at[idx, "q_mvar"] = q
        else:
            # Should not happen with current meta
            continue


def main():
    base = Path(__file__).resolve().parents[1]
    proc = base / "processed"

    # Load network
    net = pp.from_json(str(base / "crete2030_net.json"))

    # Load mapping meta and synthetic day
    meta = pd.read_csv(proc / "timeseries_meta.csv")
    syn = np.load(proc / "synthetic_day_001.npz", allow_pickle=True)

    t_syn = syn["t"]          # (M,)
    P_syn = syn["P"]          # (M, N)
    Q_syn = syn["Q"]          # (M, N)
    columns = syn["columns"]  # (N,)

    M, N = P_syn.shape
    assert Q_syn.shape == (M, N)
    assert len(meta) == N

    # Basic sanity: column order in synthetic matches meta
    if not np.all(meta["column"].values == columns.astype(str)):
        print("WARNING: meta column order differs from synthetic columns")

    # Arrays for results
    converged = np.zeros(M, dtype=bool)

    vmin = np.full(M, np.nan)
    vmax = np.full(M, np.nan)
    n_v_viol = np.zeros(M, dtype=int)

    line_max = np.full(M, np.nan)
    n_line_over = np.zeros(M, dtype=int)

    trafo_max = np.full(M, np.nan)
    n_trafo_over = np.zeros(M, dtype=int)

    # Voltage and loading limits
    v_min_lim = 0.9
    v_max_lim = 1.1
    loading_lim = 100.0  # percent

    for k in range(M):
        P_t = P_syn[k, :]
        Q_t = Q_syn[k, :]

        apply_snapshot_to_net(net, meta, P_t, Q_t)

        try:
            # AC load flow, warm-start from previous results
            pp.runpp(net, algorithm="nr", init="results", numba=False)
        except Exception as e:
            converged[k] = False
            continue

        if not net.converged:
            converged[k] = False
            continue

        converged[k] = True

        # Voltages
        vm = net.res_bus.vm_pu.to_numpy()
        vmin[k] = float(np.min(vm))
        vmax[k] = float(np.max(vm))
        n_v_viol[k] = int(np.sum((vm < v_min_lim) | (vm > v_max_lim)))

        # Lines
        if len(net.line) > 0 and "loading_percent" in net.res_line:
            lp = net.res_line.loading_percent.to_numpy()
            line_max[k] = float(np.max(lp))
            n_line_over[k] = int(np.sum(lp > loading_lim))

        # Transformers
        if len(net.trafo) > 0 and "loading_percent" in net.res_trafo:
            tp = net.res_trafo.loading_percent.to_numpy()
            trafo_max[k] = float(np.max(tp))
            n_trafo_over[k] = int(np.sum(tp > loading_lim))

    # Summary
    conv_rate = float(np.mean(converged))
    print(f"Feasible (converged) time steps: {np.sum(converged)}/{M} ({conv_rate * 100:.1f}%)")

    if np.any(converged):
        print("Voltage min / max over converged steps:",
              np.nanmin(vmin[converged]), np.nanmax(vmax[converged]))
        print("Max line loading over converged steps:",
              np.nanmax(line_max[converged]) if np.any(~np.isnan(line_max[converged])) else "n/a")
        print("Max trafo loading over converged steps:",
              np.nanmax(trafo_max[converged]) if np.any(~np.isnan(trafo_max[converged])) else "n/a")

    # Save detailed results
    out_path = proc / "synthetic_day_001_pf_results.npz"
    np.savez(
        out_path,
        t=t_syn,
        converged=converged,
        vmin=vmin,
        vmax=vmax,
        n_v_viol=n_v_viol,
        line_max=line_max,
        n_line_over=n_line_over,
        trafo_max=trafo_max,
        n_trafo_over=n_trafo_over,
        v_min_lim=v_min_lim,
        v_max_lim=v_max_lim,
        loading_lim=loading_lim,
    )
    print("Saved PF results to:", out_path)


if __name__ == "__main__":
    main()
