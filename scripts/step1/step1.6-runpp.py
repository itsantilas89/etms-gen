# step1.6-runpp.py
from pathlib import Path
import pandapower as pp

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
    net = pp.from_json(ROOT / "crete2030_net.json")

    pp.runpp(net, algorithm="nr")  # Newton-Raphson

    print("converged:", net.converged)
    print("\nBus voltages (first 10):")
    print(net.res_bus[["vm_pu", "va_degree"]].head(10))

    print("\nLine loading (first 10):")
    print(net.res_line[["loading_percent"]].head(10))


if __name__ == "__main__":
    main()
