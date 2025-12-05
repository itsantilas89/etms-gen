# step1.0-import_cgmes.py
from pathlib import Path

import pandapower as pp
from pandapower.converter.cim import from_cim as cim2pp

ROOT = Path(__file__).resolve().parents[2]

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
    files = [
        str(XML_DIR / "Equipement-EQ.xml"),
        str(XML_DIR / "Topology-TP.xml"),
        str(XML_DIR / "DiagramLayout-DL.xml"),
        str(XML_DIR / "SteadyStateHypothesis-SSH.xml"),
        str(XML_DIR / "StateValues-SV.xml"),
        str(XML_DIR / "GeographicalLocation-GL.xml"),
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

    pp.to_json(net, ROOT / "crete2030_net.json")


if __name__ == "__main__":
    main()
