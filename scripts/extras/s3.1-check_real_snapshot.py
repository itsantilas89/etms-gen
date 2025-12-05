from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp


SNAPSHOT_INDEX = 0  # change manually

# simple global limits (choose what you want)
V_MIN_LIM = 0.90
V_MAX_LIM = 1.10
LOADING_LIM = 100.0  # percent


def apply_snapshot_to_net(net, meta: pd.DataFrame, P_t: np.ndarray, Q_t: np.ndarray):
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


def main():
    base = Path(__file__).resolve().parents[1]
    proc = base / "processed"

    net = pp.from_json(str(base / "crete2030_net_v2.json"))

    data = np.load(proc / "timeseries_mapped.npz", allow_pickle=True)
    t = data["t"]
    P = data["P"]
    Q = data["Q"]
    columns = data["columns"]

    meta = pd.read_csv(proc / "timeseries_meta.csv")

    T, N = P.shape
    assert Q.shape == (T, N)
    assert len(meta) == N

    if not np.array_equal(meta["column"].astype(str).to_numpy(), columns.astype(str)):
        print("WARNING: meta['column'] and timeseries_mapped['columns'] differ in order")

    k = SNAPSHOT_INDEX
    if not (0 <= k < T):
        raise ValueError(f"SNAPSHOT_INDEX {k} out of range [0, {T-1}]")

    timestamp = pd.to_datetime(t[k])
    P_k = P[k, :]
    Q_k = Q[k, :]

    print(f"Using snapshot index {k} at time {timestamp}")

    apply_snapshot_to_net(net, meta, P_k, Q_k)

    try:
        pp.runpp(net, algorithm="nr", run_control=False)
    except Exception as e:
        print("runpp raised exception:", repr(e))
        return

    print("converged:", net.converged)
    if not net.converged:
        return

    vm = net.res_bus.vm_pu.to_numpy()
    vmin = vm.min()
    vmax = vm.max()
    n_buses = vm.size
    n_v_viol = int(np.sum((vm < V_MIN_LIM) | (vm > V_MAX_LIM)))

    print(f"Bus voltage min / max: {vmin:.4f} pu / {vmax:.4f} pu")
    print(f"Bus voltage violations: {n_v_viol} / {n_buses}")

    v_ok = (vmin >= V_MIN_LIM) and (vmax <= V_MAX_LIM)

    if len(net.line) and "loading_percent" in net.res_line:
        lp = net.res_line.loading_percent.to_numpy()
        line_max = lp.max()
        n_line_over = int(np.sum(lp > LOADING_LIM))
        print(f"Max line loading: {line_max:.2f} %  (violations: {n_line_over} / {lp.size})")
    else:
        lp = np.array([])
        line_max = np.nan
        n_line_over = 0
        print("No line loading results.")

    if len(net.trafo) and "loading_percent" in net.res_trafo:
        tp = net.res_trafo.loading_percent.to_numpy()
        trafo_max = tp.max()
        n_trafo_over = int(np.sum(tp > LOADING_LIM))
        print(f"Max trafo loading: {trafo_max:.2f} % (violations: {n_trafo_over} / {tp.size})")
    else:
        tp = np.array([])
        trafo_max = np.nan
        n_trafo_over = 0
        print("No trafo loading results.")

    line_ok = (lp.size == 0) or np.all(lp <= LOADING_LIM)
    trafo_ok = (tp.size == 0) or np.all(tp <= LOADING_LIM)

    all_ok = v_ok and line_ok and trafo_ok

    print("\nLimit check:")
    print(f"  Voltage within [{V_MIN_LIM}, {V_MAX_LIM}] pu : {v_ok}")
    print(f"  Lines <= {LOADING_LIM:.1f}%                  : {line_ok}")
    print(f"  Trafos <= {LOADING_LIM:.1f}%                 : {trafo_ok}")
    print(f"  ALL LIMITS OK                                : {all_ok}")


if __name__ == "__main__":
    main()
