# step1.2-validate_import.py
from pathlib import Path
import xml.etree.ElementTree as ET

import pandapower as pp


# CIM namespace
NS = {"cim": "http://iec.ch/TC57/2013/CIM-schema-cim16#"}

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


def count(root, tag: str) -> int:
    """Count CIM elements of a given class name in the XML tree."""
    return len(root.findall(f".//cim:{tag}", NS))


def main():
    # Paths
    eq_path = XML_DIR / "Equipement-EQ.xml"

    net_path = ROOT / "crete2030_net_v2.json"

    # Load XML and pandapower net
    eq_root = ET.parse(eq_path).getroot()
    net = pp.from_json(str(net_path))

    print("=== CGMES vs pandapower element counts ===")

    # Lines
    n_acline = count(eq_root, "ACLineSegment")
    print(f"ACLineSegment (EQ)      : {n_acline:5d}  -> net.line        : {len(net.line):5d}")

    # Transformers
    n_trf = count(eq_root, "PowerTransformer")
    print(f"PowerTransformer (EQ)   : {n_trf:5d}  -> net.trafo       : {len(net.trafo):5d}")

    # Loads (concrete subclasses)
    n_conf = count(eq_root, "ConformLoad")
    n_nonconf = count(eq_root, "NonConformLoad")
    n_stsup = count(eq_root, "StationSupply")
    n_load_total = n_conf + n_nonconf + n_stsup
    print(
        f"Loads (CIM, concrete)   : {n_load_total:5d} "
        f"(Conform={n_conf}, NonConform={n_nonconf}, Station={n_stsup})"
        f"  -> net.load        : {len(net.load):5d}"
    )

    # Generator-like equipment (mapped to gen / sgen)
    n_syn = count(eq_root, "SynchronousMachine")
    n_async = count(eq_root, "AsynchronousMachine")
    n_esrc = count(eq_root, "EnergySource")
    n_gen_like = n_syn + n_async + n_esrc
    print(
        f"Gen-like equip (CIM)    : {n_gen_like:5d} "
        f"(Syn={n_syn}, Async={n_async}, ESource={n_esrc})"
        f"  -> net.gen+net.sgen : {len(net.gen) + len(net.sgen):5d}"
    )

    # Shunts / compensators
    n_lsh = count(eq_root, "LinearShuntCompensator")
    n_nlsh = count(eq_root, "NonlinearShuntCompensator")
    n_sh_total = n_lsh + n_nlsh
    print(
        f"Shunts (CIM)            : {n_sh_total:5d} "
        f"(Linear={n_lsh}, Nonlinear={n_nlsh})"
        f"  -> net.shunt       : {len(net.shunt):5d}"
    )

    # Switches (optional)
    n_brk = count(eq_root, "Breaker")
    n_disc = count(eq_root, "Disconnector")
    n_sw = count(eq_root, "Switch")
    n_lbsw = count(eq_root, "LoadBreakSwitch")
    n_sw_total = n_brk + n_disc + n_sw + n_lbsw
    if n_sw_total > 0:
        print(
            f"Switches (CIM)          : {n_sw_total:5d} "
            f"(Brk={n_brk}, Disc={n_disc}, Sw={n_sw}, LBSw={n_lbsw})"
            f"  -> net.switch      : {len(net.switch):5d}"
        )

    print("\n=== origin_id sanity ===")
    for name, df in [
        ("bus", net.bus),
        ("line", net.line),
        ("trafo", net.trafo),
        ("load", net.load),
        ("gen", net.gen),
        ("sgen", net.sgen),
        ("shunt", net.shunt),
    ]:
        has_origin = "origin_id" in df.columns
        print(f"{name:6s}: rows={len(df):5d}, origin_id column: {has_origin}")

    # Controller presence and basic introspection
    print("\n=== controllers ===")
    if hasattr(net, "controller") and len(net.controller):
        print("controller table columns:", list(net.controller.columns))
        print("first 5 controllers:")
        print(net.controller.head())
    else:
        print("No controllers present.")


if __name__ == "__main__":
    main()
