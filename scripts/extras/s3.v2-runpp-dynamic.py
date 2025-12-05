from pathlib import Path
import pandapower as pp


def main():
    base = Path(__file__).resolve().parent

    # Use the CGMES-based net created by s1.1
    net_path = base / ".." / "crete2030_net_v2.json"
    net = pp.from_json(str(net_path))

    # Plain AC power flow, no controllers
    pp.runpp(net, algorithm="nr", run_control=False)

    print("converged:", net.converged)

    print("\nBus voltages (first 10):")
    if len(net.res_bus):
        print(net.res_bus[["vm_pu", "va_degree"]].head(10))
    else:
        print("(no bus results)")

    print("\nLine loading (first 10):")
    if len(net.line) and len(net.res_line):
        print(net.res_line[["loading_percent"]].head(10))
    else:
        print("(no lines or no line results)")


if __name__ == "__main__":
    main()
