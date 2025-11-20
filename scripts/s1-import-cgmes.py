#s1-import-cgmes.py
from pathlib import Path

import pandapower as pp
from pandapower.converter.cim import from_cim as cim2pp


def main():
    base = Path(__file__).resolve().parent
    cgmes = base / ".."

    files = [
        str(cgmes / "Equipement-EQ.xml"),
        str(cgmes / "Topology-TP.xml"),
        str(cgmes / "DiagramLayout-DL.xml"),
        str(cgmes / "SteadyStateHypothesis-SSH.xml"),
        str(cgmes / "StateValues-SV.xml"),
        str(cgmes / "GeographicalLocation-GL.xml"),
    ]

    net = cim2pp.from_cim(
        file_list=files,
        cgmes_version="2.4.15",
        run_powerflow=False,
        ignore_errors=False,
    )

    print("buses:", len(net.bus))
    print("lines:", len(net.line))
    print("trafos:", len(net.trafo))
    print("loads:", len(net.load))

    pp.to_json(net, base / ... / "Crete 2030" / "crete2030_net.json")


if __name__ == "__main__":
    main()
