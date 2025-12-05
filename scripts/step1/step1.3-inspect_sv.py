# step1.3-inspect_sv.py
from pathlib import Path
import xml.etree.ElementTree as ET

import pandapower as pp


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
    return len(root.findall(f".//cim:{tag}", NS))


def main():
    sv_path = XML_DIR / "StateValues-SV.xml"

    net_path = ROOT / ".." / "crete2030_net_v2.json"

    sv_root = ET.parse(sv_path).getroot()
    net = pp.from_json(str(net_path))

    print("=== StateVariables (SV) coverage ===")
    n_sv_v = count(sv_root, "SvVoltage")
    n_sv_pf = count(sv_root, "SvPowerFlow")
    n_sv_tap = count(sv_root, "SvTapStep")
    n_sv_sh = count(sv_root, "SvShuntCompensatorSections")

    print(f"SvVoltage entries                : {n_sv_v}")
    print(f"SvPowerFlow entries             : {n_sv_pf}")
    print(f"SvTapStep entries               : {n_sv_tap}")
    print(f"SvShuntCompensatorSections      : {n_sv_sh}")

    print("\n=== Pandapower net – elements with state == from CGMES ===")
    print(f"buses   : {len(net.bus)}")
    print(f"trafos  : {len(net.trafo)}")
    print(f"shunts  : {len(net.shunt)}")
    print(f"lines   : {len(net.line)}")

    # Show tap-related info for first few trafos
    if len(net.trafo):
        cols = [c for c in ["name", "vn_hv_kv", "vn_lv_kv",
                            "tap_pos", "tap_min", "tap_max",
                            "tap_step_percent", "tap_step_degree",
                            "origin_id"] if c in net.trafo.columns]
        print("\nFirst 10 trafos (tap-related columns):")
        print(net.trafo[cols].head(10))
    else:
        print("\nNo trafos in net.")

    # Show shunt steps
    if len(net.shunt):
        cols = [c for c in ["name", "bus", "q_mvar", "p_mw",
                            "step", "max_step", "origin_id"]
                if c in net.shunt.columns]
        print("\nAll shunts (step-related columns):")
        print(net.shunt[cols])
    else:
        print("\nNo shunts in net.")

    # Controllers info (tap controllers created from CGMES)
    print("\n=== Controllers in net ===")
    if hasattr(net, "controller") and len(net.controller):
        print("controller table columns:", list(net.controller.columns))
        print("first 10 controllers:")
        print(net.controller.head(10))
    else:
        print("No controllers present.")

    # Optional: quick PF run with controllers, just to see res tables exist
    print("\n=== Quick PF run with controllers (no checks, just to populate results) ===")
    try:
        pp.runpp(net, algorithm="nr", run_control=True)
        print("Power flow converged:", net.converged)
        print("Sample bus results (first 10):")
        print(net.res_bus[["vm_pu", "va_degree"]].head(10))
    except Exception as e:
        print("Power flow failed with exception:", repr(e))


if __name__ == "__main__":
    main()
