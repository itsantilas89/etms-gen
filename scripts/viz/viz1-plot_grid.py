# viz1-plot_grid.py
from pathlib import Path

import pandapower as pp
from pandapower.plotting import simple_plot
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
    net_path = ROOT / "crete2030_net.json"

    net = pp.from_json(str(net_path))

    simple_plot(
        net,
        plot_loads=True,   # draw load symbols at the buses
        plot_sgens=True,   # draw static generator symbols
        load_size=1.5,
        sgen_size=1.5,
        bus_size=1.0,
    )
    plt.show()


if __name__ == "__main__":
    main()
