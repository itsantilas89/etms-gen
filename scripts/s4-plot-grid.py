from pathlib import Path

import pandapower as pp
from pandapower.plotting import simple_plot
import matplotlib.pyplot as plt


def main():
    base = Path(__file__).resolve().parent
    net_path = base / ".." / "crete2030_net.json"

    # cast to str so pandapower treats it as filename, not JSON text
    net = pp.from_json(str(net_path))

    simple_plot(net)
    plt.show()


if __name__ == "__main__":
    main()
