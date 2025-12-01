#s4.1-plot-grid.py
from pathlib import Path

import pandapower as pp
from pandapower.plotting import simple_plot
import matplotlib
import matplotlib.pyplot as plt


def main():
    # Force non-interactive backend (safe on SSH)
    matplotlib.use("Agg")

    base = Path(__file__).resolve().parent
    net_path = base / ".." / "crete2030_net.json"

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

    out_path = base / ".." / "grid_topology.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved grid plot to: {out_path}")


if __name__ == "__main__":
    main()
