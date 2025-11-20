from pathlib import Path
import pandapower as pp

def main():
    base = Path(__file__).resolve().parent
    net_path = base / ".." / "crete2030_net.json"

    net = pp.from_json(net_path)

    print("net name:", net.name if hasattr(net, "name") else None)
    print("buses   :", len(net.bus))
    print("lines   :", len(net.line))
    print("trafos  :", len(net.trafo))
    print("loads   :", len(net.load))
    print("gens    :", len(net.gen))
    print("sgen    :", len(net.sgen))
    print("shunts  :", len(net.shunt))

    print("\nFirst 10 buses:")
    print(net.bus[["name", "vn_kv"]].head(10))

    print("\nFirst 10 lines:")
    print(net.line[["name", "from_bus", "to_bus", "length_km"]].head(10))

    print("\nFirst 10 loads:")
    print(net.load[["name", "bus", "p_mw", "q_mvar"]].head(10))


if __name__ == "__main__":
    main()
