# step1.1-import_cgmes_dynamic.py
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
        # use SV profile to create measurements (for later validation / debugging)
        create_measurements="SV",
        # create tap controllers for transformer tap changers (OLTC behaviour)
        create_tap_controller=True,
        # optional: import all diagrams; harmless for your use case
        diagram_name="all",
        # do NOT run power flow inside the converter; you run it explicitly later
        run_powerflow=False,
        # fail loudly if conversion has internal errors
        ignore_errors=False,
    )

    print("Imported CGMES into pandapower net")
    print("  buses :", len(net.bus))
    print("  lines :", len(net.line))
    print("  trafos:", len(net.trafo))
    print("  loads :", len(net.load))
    print("  gens  :", len(net.gen))
    print("  sgens :", len(net.sgen))
    print("  shunts:", len(net.shunt))
    print("  controllers:", len(net.controller) if hasattr(net, "controller") else 0)
    if "measurement" in net:
        print("  measurements:", len(net.measurement))

    # Save to JSON
    out_path = ROOT / "crete2030_net_v2.json"
    pp.to_json(net, out_path)
    print("Saved net to:", out_path)


if __name__ == "__main__":
    main()
