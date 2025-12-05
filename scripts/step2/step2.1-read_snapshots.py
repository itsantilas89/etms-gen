# step2.1-read_snapshots.py
from pathlib import Path

import pandas as pd


def flatten_columns(columns):
    """
    Flatten a 3-level MultiIndex of the form:
        (substation, category, equipment_id)
    to a single string.

    For TIMESTAMP, only the top level is used.
    For others, use the equipment_id (third level).
    """
    flat = []
    for col in columns:
        # col can be a tuple (substation, category, equipment_id)
        # or a plain string if header isn't multi-level
        if not isinstance(col, tuple):
            flat.append(str(col))
            continue

        top, mid, low = col

        # TIMESTAMP column: top = "TIMESTAMP", others usually NaN/None
        if isinstance(top, str) and top.upper() == "TIMESTAMP":
            flat.append("TIMESTAMP")
            continue

        # Prefer the equipment_id in the third level if present
        if isinstance(low, str) and low.strip() != "":
            flat.append(low.strip())
            continue

        # Fallback: substation name or category if low is missing
        if isinstance(mid, str) and mid.strip() != "":
            flat.append(mid.strip())
        elif isinstance(top, str) and top.strip() != "":
            flat.append(top.strip())
        else:
            flat.append("_".join(str(x) for x in col if pd.notna(x)))

    return flat


def load_snapshot_excel(path: Path, p_sheet: str, q_sheet: str):
    """
    Load one day's Excel file with two sheets (P and Q).

    Returns:
        p_df: DataFrame with columns ['TIMESTAMP', <equipment_ids...>] in MW
        q_df: DataFrame with same columns layout in MVAr
    """
    # Read with three header rows -> MultiIndex columns
    p_raw = pd.read_excel(path, sheet_name=p_sheet, header=[0, 1, 2])
    q_raw = pd.read_excel(path, sheet_name=q_sheet, header=[0, 1, 2])

    # Flatten multi-level headers
    p_raw.columns = flatten_columns(p_raw.columns)
    q_raw.columns = flatten_columns(q_raw.columns)

    # Ensure TIMESTAMP column is datetime
    if "TIMESTAMP" in p_raw.columns:
        p_raw["TIMESTAMP"] = pd.to_datetime(p_raw["TIMESTAMP"])
    if "TIMESTAMP" in q_raw.columns:
        q_raw["TIMESTAMP"] = pd.to_datetime(q_raw["TIMESTAMP"])

    return p_raw, q_raw


def main():
    ROOT = Path(__file__).resolve().parents[2]

    EXCEL_SNAPSHOT_DIR = ROOT / "excel_snapshots"

    day1_file = EXCEL_SNAPSHOT_DIR / "PowerProfilesData-Jan10.xlsx"
    day2_file = EXCEL_SNAPSHOT_DIR / "PowerProfilesData-Jun10.xlsx"

    p_sheet_name = "active power (MW)"
    q_sheet_name = "reactive power (MVAR)"
    
    print("Loading:", day1_file)
    p1, q1 = load_snapshot_excel(day1_file, p_sheet_name, q_sheet_name)
    print("Loading:", day2_file)
    p2, q2 = load_snapshot_excel(day2_file, p_sheet_name, q_sheet_name)

    # Drop rows without timestamp (extra header/footer)
    for df_name, df in [("p1", p1), ("p2", p2), ("q1", q1), ("q2", q2)]:
        if "TIMESTAMP" in df.columns:
            before = len(df)
            df.dropna(subset=["TIMESTAMP"], inplace=True)
            after = len(df)
            print(f"{df_name}: dropped {before - after} rows without TIMESTAMP")

    # Columns must match per P/Q group
    assert p1.columns.tolist() == p2.columns.tolist()
    assert q1.columns.tolist() == q2.columns.tolist()

    p_all = pd.concat([p1, p2], ignore_index=True)
    q_all = pd.concat([q1, q2], ignore_index=True)

    # Remove Unnamed columns (we don't need them for mapping)
    p_cols_keep = [c for c in p_all.columns if c == "TIMESTAMP" or not str(c).startswith("Unnamed")]
    q_cols_keep = [c for c in q_all.columns if c == "TIMESTAMP" or not str(c).startswith("Unnamed")]
    p_all = p_all[p_cols_keep]
    q_all = q_all[q_cols_keep]

    # Align P and Q on common timestamps
    common_ts = sorted(set(p_all["TIMESTAMP"]).intersection(q_all["TIMESTAMP"]))
    p_all = p_all[p_all["TIMESTAMP"].isin(common_ts)].reset_index(drop=True)
    q_all = q_all[q_all["TIMESTAMP"].isin(common_ts)].reset_index(drop=True)

    print("P_all shape:", p_all.shape)
    print("Q_all shape:", q_all.shape)

    out_dir = ROOT / "processed"
    out_dir.mkdir(exist_ok=True)

    p_all.to_parquet(out_dir / "P_all.parquet")
    q_all.to_parquet(out_dir / "Q_all.parquet")

    print("\nSaved aligned P/Q to 'processed' directory.")


if __name__ == "__main__":
    main()
