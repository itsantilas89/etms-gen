#s12-scale_correct_day.py
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
            net.load.at[idx, "p_mw"] = max(p, 0.0)
            net.load.at[idx, "q_mvar"] = q
        elif table == "sgen":
            net.sgen.at[idx, "p_mw"] = max(p, 0.0)
            net.sgen.at[idx, "q_mvar"] = q
        elif table == "gen":
            net.gen.at[idx, "p_mw"] = max(p, 0.0)
            net.gen.at[idx, "q_mvar"] = q
        elif table == "shunt":
            net.shunt.at[idx, "p_mw"] = 0.0
            net.shunt.at[idx, "q_mvar"] = q
        else:
            continue


def run_pf_and_check(net, v_min_lim, v_max_lim, loading_lim):
    """
    Run AC power flow and check voltage, line, trafo constraints.
    Returns (ok, vmin, vmax, line_max, trafo_max).
    """
    try:
        pp.runpp(net, algorithm="nr", init="results", numba=False)
    except Exception:
        return False, np.nan, np.nan, np.nan, np.nan

    if not net.converged:
        return False, np.nan, np.nan, np.nan, np.nan

    vm = net.res_bus.vm_pu.to_numpy()
    vmin = float(vm.min())
    vmax = float(vm.max())

    if vmin < v_min_lim or vmax > v_max_lim:
        return False, vmin, vmax, np.nan, np.nan

    line_max = np.nan
    if len(net.line) > 0 and "loading_percent" in net.res_line:
        lp = net.res_line.loading_percent.to_numpy()
        line_max = float(lp.max())
        if np.any(lp > loading_lim):
            return False, vmin, vmax, line_max, np.nan

    trafo_max = np.nan
    if len(net.trafo) > 0 and "loading_percent" in net.res_trafo:
        tp = net.res_trafo.loading_percent.to_numpy()
        trafo_max = float(tp.max())
        if np.any(tp > loading_lim):
            return False, vmin, vmax, line_max, trafo_max

    return True, vmin, vmax, line_max, trafo_max


def main():
    base = Path(__file__).resolve().parents[1]
    proc = base / "processed"

    # Load network and meta
    net = pp.from_json(str(base / "crete2030_net.json"))
    meta = pd.read_csv(proc / "timeseries_meta.csv")

    # Load synthetic day and existing PF limits
    syn = np.load(proc / "synthetic_day_001.npz", allow_pickle=True)
    pf = np.load(proc / "synthetic_day_001_pf_results.npz", allow_pickle=True)

    t_syn = syn["t"]          # (M,)
    P_syn = syn["P"]          # (M, N)
    Q_syn = syn["Q"]          # (M, N)
    columns = syn["columns"]  # (N,)

    M, N = P_syn.shape
    assert Q_syn.shape == (M, N)
    assert len(meta) == N

    # Limits
    v_min_lim = float(pf["v_min_lim"])
    v_max_lim = float(pf["v_max_lim"])
    loading_lim = float(pf["loading_lim"])

    # Arrays for corrected series and scales
    P_corr = np.zeros_like(P_syn)
    Q_corr = np.zeros_like(Q_syn)
    alpha = np.ones(M, dtype=float)

    ok_flags = np.zeros(M, dtype=bool)

    for k in range(M):
        P0 = P_syn[k, :].copy()
        Q0 = Q_syn[k, :].copy()

        # First, test alpha = 1.0 (original)
        apply_snapshot_to_net(net, meta, P0, Q0)
        ok, _, _, _, _ = run_pf_and_check(net, v_min_lim, v_max_lim, loading_lim)

        if ok:
            alpha[k] = 1.0
            P_corr[k, :] = P0
            Q_corr[k, :] = Q0
            ok_flags[k] = True
            continue

        # If not ok, binary search in alpha ∈ (0, 1]
        lo, hi = 0.0, 1.0
        best_alpha = 0.0

        for _ in range(12):  # ~1/2^12 resolution
            mid = 0.5 * (lo + hi)
            if mid <= 0.0:
                break

            P_try = mid * P0
            Q_try = mid * Q0

            apply_snapshot_to_net(net, meta, P_try, Q_try)
            ok_mid, _, _, _, _ = run_pf_and_check(net, v_min_lim, v_max_lim, loading_lim)

            if ok_mid:
                best_alpha = mid
                lo = mid
            else:
                hi = mid

        if best_alpha > 0.0:
            alpha[k] = best_alpha
            P_corr[k, :] = best_alpha * P0
            Q_corr[k, :] = best_alpha * Q0
            ok_flags[k] = True
        else:
            # Fall back to zero injections if nothing else works
            alpha[k] = 0.0
            P_corr[k, :] = 0.0
            Q_corr[k, :] = 0.0
            ok_flags[k] = False

    print(f"Time steps with some feasible alpha: {ok_flags.sum()}/{M}")

    out_path = proc / "synthetic_day_001_corrected.npz"
    np.savez(
        out_path,
        t=t_syn,
        P=P_corr,
        Q=Q_corr,
        columns=columns,
        alpha=alpha,
        feasible=ok_flags,
        v_min_lim=v_min_lim,
        v_max_lim=v_max_lim,
        loading_lim=loading_lim,
    )
    print("Saved corrected synthetic day to:", out_path)


if __name__ == "__main__":
    main()
