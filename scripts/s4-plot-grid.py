from pathlib import Path

import pandapower as pp
from pandapower.plotting import simple_plot
import matplotlib.pyplot as plt


def main():
    base = Path(__file__).resolve().parent
    net_path = base / ".." / "crete2030_net.json"

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
