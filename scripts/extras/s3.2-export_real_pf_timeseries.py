from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp


# Global limits (for summary only, not mandatory)
V_MIN_LIM = 0.90
V_MAX_LIM = 1.10
LOADING_LIM = 100.0  # percent


def apply_snapshot_to_net(net, meta: pd.DataFrame, P_t: np.ndarray, Q_t: np.ndarray):
    """
    Write one time-step (P_t, Q_t) into the pandapower net according to meta.
    P_t, Q_t are 1D arrays of length N (same column order as meta['column']).
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


def main():
    base = Path(__file__).resolve().parents[1]
    proc = base / "processed"
    out_dir = proc / "real_pf_exports"
    out_dir.mkdir(exist_ok=True)

    # 1) Load frozen CGMES-based network
    net = pp.from_json(str(base / "crete2030_net_v2.json"))

    # 2) Load real time series and meta mapping
    data = np.load(proc / "timeseries_mapped.npz", allow_pickle=True)
    t = data["t"]              # (T,)
    P = data["P"]              # (T, N)
    Q = data["Q"]              # (T, N)
    columns = data["columns"]  # (N,)

    meta = pd.read_csv(proc / "timeseries_meta.csv")

    T, N = P.shape
    assert Q.shape == (T, N)
    assert len(meta) == N

    if not np.array_equal(meta["column"].astype(str).to_numpy(), columns.astype(str)):
        print("WARNING: meta['column'] and timeseries_mapped['columns'] differ in order")

    # 3) Static info for buses and lines (names, connections)
    # Let reset_index create the index column once and rename it.
    bus_static = net.bus.copy()
    bus_static = bus_static.reset_index().rename(columns={"index": "bus_idx"})

    line_static = net.line.copy()
    line_static = line_static.reset_index().rename(columns={"index": "line_idx"})

    # 4) Storage for results
    bus_results = []
    line_results = []
    summary_rows = []

    # 5) Loop over all time steps
    for k in range(T):
        timestamp = pd.to_datetime(t[k])
        P_k = P[k, :]
        Q_k = Q[k, :]

        apply_snapshot_to_net(net, meta, P_k, Q_k)

        try:
            # Frozen controls: taps/shunts fixed as imported
            pp.runpp(net, algorithm="nr", init="results", numba=False, run_control=False)
            conv = bool(net.converged)
        except Exception as e:
            print(f"[{k}/{T}] {timestamp} runpp exception: {repr(e)}")
            conv = False

        if not conv:
            # Record summary with convergence failure, no detailed bus/line results
            summary_rows.append({
                "timestamp": timestamp,
                "index": k,
                "converged": False,
                "vmin_pu": np.nan,
                "vmax_pu": np.nan,
                "n_v_viol": np.nan,
                "line_max_loading_percent": np.nan,
                "n_line_over": np.nan,
                "trafo_max_loading_percent": np.nan,
                "n_trafo_over": np.nan,
            })
            continue

        # Bus voltages
        vm = net.res_bus.vm_pu.to_numpy()
        vmin = float(vm.min())
        vmax = float(vm.max())
        n_buses = vm.size
        n_v_viol = int(np.sum((vm < V_MIN_LIM) | (vm > V_MAX_LIM)))

        # Line loadings
        if len(net.line) and "loading_percent" in net.res_line:
            lp = net.res_line.loading_percent.to_numpy()
            line_max = float(lp.max())
            n_line_over = int(np.sum(lp > LOADING_LIM))
        else:
            lp = np.array([])
            line_max = np.nan
            n_line_over = 0

        # Trafo loadings
        if len(net.trafo) and "loading_percent" in net.res_trafo:
            tp = net.res_trafo.loading_percent.to_numpy()
            trafo_max = float(tp.max())
            n_trafo_over = int(np.sum(tp > LOADING_LIM))
        else:
            tp = np.array([])
            trafo_max = np.nan
            n_trafo_over = 0

        summary_rows.append({
            "timestamp": timestamp,
            "index": k,
            "converged": True,
            "vmin_pu": vmin,
            "vmax_pu": vmax,
            "n_buses": n_buses,
            "n_v_viol": n_v_viol,
            "line_max_loading_percent": line_max,
            "n_line_over": n_line_over,
            "trafo_max_loading_percent": trafo_max,
            "n_trafo_over": n_trafo_over,
        })

        # Detailed bus results
        br = net.res_bus[["vm_pu", "va_degree"]].copy()
        br["timestamp"] = timestamp
        br["time_index"] = k
        br["bus_idx"] = br.index
        bus_results.append(br)

        # Detailed line results (flows + loading)
        if len(net.line) and len(net.res_line):
            lr = net.res_line[
                [
                    "p_from_mw", "q_from_mvar",
                    "p_to_mw", "q_to_mvar",
                    "loading_percent",
                    "vm_from_pu", "va_from_degree",
                    "vm_to_pu", "va_to_degree",
                ]
            ].copy()
            lr["timestamp"] = timestamp
            lr["time_index"] = k
            lr["line_idx"] = lr.index
            line_results.append(lr)

        if (k + 1) % 100 == 0 or k == T - 1:
            print(f"[{k+1}/{T}] processed up to {timestamp}, converged={conv}")

    # 6) Concatenate and merge static info, then export CSVs

    # Summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "real_pf_summary.csv", index=False)
    print("Saved summary to:", out_dir / "real_pf_summary.csv")

    # Bus results
    if bus_results:
        bus_results_df = pd.concat(bus_results, ignore_index=True)
        # merge on bus_idx directly; bus_static already has bus_idx column
        bus_results_df = bus_results_df.merge(
            bus_static,
            on="bus_idx",
            how="left",
            suffixes=("", "_bus"),
        )
        # Keep a reasonable column order
        bus_cols = [
            "timestamp", "time_index", "bus_idx",
            "name", "vn_kv",
            "vm_pu", "va_degree",
        ]
        bus_cols = [c for c in bus_cols if c in bus_results_df.columns]
        bus_results_df[bus_cols].to_csv(out_dir / "real_pf_bus_results.csv", index=False)
        print("Saved bus results to:", out_dir / "real_pf_bus_results.csv")

    # Line results
    if line_results:
        line_results_df = pd.concat(line_results, ignore_index=True)
        line_results_df = line_results_df.merge(
            line_static,
            on="line_idx",
            how="left",
            suffixes=("", "_line"),
        )
        line_cols = [
            "timestamp", "time_index", "line_idx",
            "name", "from_bus", "to_bus", "length_km",
            "p_from_mw", "q_from_mvar",
            "p_to_mw", "q_to_mvar",
            "loading_percent",
            "vm_from_pu", "va_from_degree",
            "vm_to_pu", "va_to_degree",
        ]
        line_cols = [c for c in line_cols if c in line_results_df.columns]
        line_results_df[line_cols].to_csv(out_dir / "real_pf_line_results.csv", index=False)
        print("Saved line results to:", out_dir / "real_pf_line_results.csv")


if __name__ == "__main__":
    main()
