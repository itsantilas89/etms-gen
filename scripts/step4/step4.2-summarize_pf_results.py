# step4.2-summarize_pf_results.py
from pathlib import Path
import numpy as np


def main():
    base = Path(__file__).resolve().parents[1]
    proc = base / "processed"

    pf = np.load(proc / pf / "day001_trans_pf_results.npz", allow_pickle=True)

    converged = pf["converged"]           # (M,)
    vmin = pf["vmin"]
    vmax = pf["vmax"]
    line_max = pf["line_max"]
    trafo_max = pf["trafo_max"]
    v_min_lim = float(pf["v_min_lim"])
    v_max_lim = float(pf["v_max_lim"])
    loading_lim = float(pf["loading_lim"])

    M = len(converged)

    feas_conv = converged
    feas_v = (vmin >= v_min_lim) & (vmax <= v_max_lim)

    # Handle NaNs: if there are no lines or trafos, treat as ok
    feas_line = np.ones(M, dtype=bool)
    if np.any(~np.isnan(line_max)):
        feas_line = line_max <= loading_lim

    feas_trafo = np.ones(M, dtype=bool)
    if np.any(~np.isnan(trafo_max)):
        feas_trafo = trafo_max <= loading_lim

    feas_all = feas_conv & feas_v & feas_line & feas_trafo

    print(f"Total steps: {M}")
    print(f"Converged only     : {feas_conv.sum()}/{M}")
    print(f"Voltage OK         : {feas_v.sum()}/{M}")
    print(f"Line loading OK    : {feas_line.sum()}/{M}")
    print(f"Trafo loading OK   : {feas_trafo.sum()}/{M}")
    print(f"All constraints OK : {feas_all.sum()}/{M}")

    if feas_all.any():
        print("Min vmin over fully-feasible:", float(vmin[feas_all].min()))
        print("Max vmax over fully-feasible:", float(vmax[feas_all].max()))
        print("Max line loading over fully-feasible:",
              float(line_max[feas_all].max()) if np.any(~np.isnan(line_max[feas_all])) else "n/a")
        print("Max trafo loading over fully-feasible:",
              float(trafo_max[feas_all].max()) if np.any(~np.isnan(trafo_max[feas_all])) else "n/a")


if __name__ == "__main__":
    main()
