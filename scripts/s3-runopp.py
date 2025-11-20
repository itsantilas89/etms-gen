#s3-runpp.py
from pathlib import Path
import pandapower as pp

def main():
    base = Path(__file__).resolve().parent
    net = pp.from_json(base / ".." / "crete2030_net.json")

    pp.runpp(net, algorithm="nr")  # Newton-Raphson

    print("converged:", net.converged)
    print("\nBus voltages (first 10):")
    print(net.res_bus[["vm_pu", "va_degree"]].head(10))

    print("\nLine loading (first 10):")
    print(net.res_line[["loading_percent"]].head(10))


if __name__ == "__main__":
    main()
