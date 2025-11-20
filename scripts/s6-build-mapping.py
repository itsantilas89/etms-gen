#s6-build-mapping.py
from pathlib import Path
import json
import re

import pandapower as pp
import pandas as pd


def build_bus_id_index_map(net):
    """
    Map numeric bus ID string (e.g. '291231') -> bus index,
    assuming bus.name starts with that ID, e.g. '291231 AG.VARV MV'.
    """
    bus_id_to_idx = {}
    for idx, name in net.bus["name"].items():
        if not isinstance(name, str):
            continue
        first = name.split()[0]
        if first.isdigit():
            bus_id_to_idx[first] = idx
    return bus_id_to_idx


def extract_bus_id(col: str):
    """
    From column names like:
      'load_291231_1_MV'
      'machine_291231_W3_MV'
      'fixed shunt_291231_1_MV'
      'switched_shunt_61833_150kV'
    extract the numeric bus id ('291231', '61833', ...).
    """
    # quick: take first number sequence in the string
    m = re.search(r"(\d+)", col)
    if not m:
        return None
    return m.group(1)


def main():
    base = Path(__file__).resolve().parents[1]

    # Load network and time series
    net = pp.from_json(str(base / "crete2030_net.json"))
    p_all = pd.read_parquet(base / "processed" / "P_all.parquet")
    q_all = pd.read_parquet(base / "processed" / "Q_all.parquet")

    # Columns to map: ignore TIMESTAMP and Unnamed
    cols = [
        c for c in p_all.columns
        if c != "TIMESTAMP" and not str(c).startswith("Unnamed")
    ]

    print("Total equipment-like columns in P_all:", len(cols))

    bus_id_to_idx = build_bus_id_index_map(net)

    mapping = {}   # excel_name -> (table, index)
    unmatched = []

    for col in cols:
        col_str = str(col)

        # Determine type from prefix
        lower = col_str.lower()

        bus_id = extract_bus_id(col_str)
        if bus_id is None or bus_id not in bus_id_to_idx:
            unmatched.append(col_str)
            continue

        bus_idx = bus_id_to_idx[bus_id]
        mapped = False

        # LOADS
        if lower.startswith("load_"):
            idxs = net.load.index[net.load.bus == bus_idx].tolist()
            if idxs:
                mapping[col_str] = ("load", int(idxs[0]))
                mapped = True

        # GENERATORS (RES PRODUCTION)
        elif lower.startswith("machine_"):
            # Prefer sgen (static generator, typical for RES)
            idxs = net.sgen.index[net.sgen.bus == bus_idx].tolist()
            if idxs:
                mapping[col_str] = ("sgen", int(idxs[0]))
                mapped = True
            else:
                # Fallback: gen table
                idxs = net.gen.index[net.gen.bus == bus_idx].tolist()
                if idxs:
                    mapping[col_str] = ("gen", int(idxs[0]))
                    mapped = True

        # FIXED SHUNTS / SWITCHED SHUNTS
        elif lower.startswith("fixed shunt_") or lower.startswith("switched_shunt_"):
            idxs = net.shunt.index[net.shunt.bus == bus_idx].tolist()
            if idxs:
                mapping[col_str] = ("shunt", int(idxs[0]))
                mapped = True

        # If not mapped by any rule
        if not mapped:
            unmatched.append(col_str)

    print("Mapped columns:", len(mapping))
    print("Unmatched columns:", len(unmatched))
    if unmatched:
        print("First 20 unmatched:", unmatched[:20])

    # Save mapping for later use
    out_path = base / "processed" / "mapping.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print("Mapping saved to:", out_path)


if __name__ == "__main__":
    main()
