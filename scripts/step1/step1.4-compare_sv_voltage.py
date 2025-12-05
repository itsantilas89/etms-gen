# step1.4-compare_sv_voltage.py
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
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


def main():
    sv_path = XML_DIR / "StateValues-SV.xml"

    net_path = ROOT / "crete2030_net_v2.json"

    # Load SV XML and pandapower net
    sv_root = ET.parse(sv_path).getroot()
    net = pp.from_json(str(net_path))

    # Collect SvVoltage data: TopologicalNode id -> (v_kv, angle_deg)
    sv_volt = {}  # topo_node_id (string without leading '#') -> (v_kv, angle_deg)

    for sv in sv_root.findall(".//cim:SvVoltage", NS):
        tn_elem = sv.find("cim:SvVoltage.TopologicalNode", NS)
        if tn_elem is None:
            continue
        ref = tn_elem.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
        if not ref:
            continue
        topo_id = ref.lstrip("#")

        v_elem = sv.find("cim:SvVoltage.v", NS)
        if v_elem is None or v_elem.text is None:
            continue
        try:
            v_kv = float(v_elem.text)
        except ValueError:
            continue

        ang_elem = sv.find("cim:SvVoltage.angle", NS)
        angle_deg = float(ang_elem.text) if (ang_elem is not None and ang_elem.text) else 0.0

        sv_volt[topo_id] = (v_kv, angle_deg)

    print(f"SvVoltage entries parsed: {len(sv_volt)}")

    # Run PF WITHOUT controllers to match the static SV snapshot
    try:
        pp.runpp(net, algorithm="nr", run_control=False)
    except Exception as e:
        print("Pandapower PF (run_control=False) raised exception:", repr(e))
        return

    if not net.converged:
        print("Pandapower PF (run_control=False) did not converge; aborting comparison.")
        return

    # Map SvVoltage TopologicalNode IDs to pandapower buses via origin_id
    bus_origin = net.bus.get("origin_id")
    if bus_origin is None:
        print("No origin_id column on bus – cannot map SvVoltage to buses.")
        return

    bus_origin = bus_origin.astype(str)

    matched_vm_pp = []
    matched_vm_sv = []

    for topo_id, (v_kv, _) in sv_volt.items():
        # naive match: origin_id string contains topo_id
        mask = bus_origin.str.contains(topo_id, na=False)
        idx = net.bus.index[mask]
        if len(idx) != 1:
            # either no match or ambiguous – skip
            continue

        bus_idx = idx[0]
        vn_kv = float(net.bus.at[bus_idx, "vn_kv"])
        if vn_kv <= 0:
            continue

        vm_pu_sv = v_kv / vn_kv
        vm_pu_pp = float(net.res_bus.at[bus_idx, "vm_pu"])

        matched_vm_sv.append(vm_pu_sv)
        matched_vm_pp.append(vm_pu_pp)

    matched_vm_sv = np.array(matched_vm_sv, dtype=float)
    matched_vm_pp = np.array(matched_vm_pp, dtype=float)

    n_match = matched_vm_sv.size
    print(f"Matched SvVoltage -> pandapower buses: {n_match}")

    if n_match == 0:
        print("No matches; adjust mapping logic if needed.")
        return

    diff = matched_vm_pp - matched_vm_sv
    abs_diff = np.abs(diff)

    print("Voltage magnitude comparison (per unit):")
    print(f"  max |Δvm|  : {abs_diff.max():.6f}")
    print(f"  mean |Δvm| : {abs_diff.mean():.6f}")
    print(f"  95th pct  : {np.percentile(abs_diff, 95):.6f}")

    print("\nFirst 10 matched samples (vm_pu_pp, vm_pu_sv, diff):")
    for i in range(min(10, n_match)):
        print(f"  {matched_vm_pp[i]:.6f}   {matched_vm_sv[i]:.6f}   {diff[i]:+.6f}")


if __name__ == "__main__":
    main()
