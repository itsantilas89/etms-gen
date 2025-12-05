# viz1.v2-plot_grid.py
from pathlib import Path

import pandapower as pp
from pandapower.plotting import simple_plot
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

XML_DIR = ROOT / "Crete_2030_xml_files"
EXCEL_SNAPSHOT_DIR = ROOT / "excel_snapshots"
PROC_DIR = ROOT / "processed"

# subfolders under processed
MODELS_DIR = PROC_DIR / "models"
STATMODEL_DIR = MODELS_DIR / "statmodel"
TRANSFORMER_DIR = MODELS_DIR / "transformer"

SYN_DIR = PROC_DIR / "synthetic"
PF_DIR = PROC_DIR / "pf"

def main():
    # Force non-interactive backend (safe on SSH)
    matplotlib.use("Agg")

    net_path = ROOT / "crete2030_net.json"

    net = pp.from_json(str(net_path))

    fig, ax = plt.subplots(figsize=(10, 8))

    simple_plot(
        net,
        plot_loads=True,
        plot_sgens=True,
        load_size=1.5,
        sgen_size=1.5,
        bus_size=1.0,
        ax=ax,
    )

    plt.tight_layout()

    out_path = ROOT / other / "grid_topology.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved grid plot to: {out_path}")


if __name__ == "__main__":
    main()
