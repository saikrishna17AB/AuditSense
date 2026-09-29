"""Loads Works_Sanctioned.csv and produces the clean schema every feature
file expects: amount, sanction_date, recommended_date, approval_gap_days,
state, agency_raw, work_raw, work_status.

Two real issues in the raw file, handled here:
1. The last row is a "Grand Total" footer (Sanction Amount reads
   "41,44,69,06,781.88" and Work Status is garbage) — dropped, not scored.
2. Dates are "09-Jul-2024" strings; amounts are strings with no separators
   (no commas in this export, but stripping them is harmless/safe anyway).
"""
import pandas as pd

DEFAULT_PATH = "Datasets/Works_Sanctioned.csv"

COLUMN_MAP = {
    "Sr. No.": "sr_no",
    "Work category": "work_category",
    "Work": "work_raw",
    "State": "state",
    "IDA": "agency_raw",
    "Hon'ble Members of Parliament": "mp_name",
    "Constituency": "constituency",
    "Work description": "work_description",
    "Recommended date": "recommended_date",
    "Sanction Date": "sanction_date",
    "Sanction Amount ( ₹ )": "amount",
    "Work Status": "work_status",
}


def load_sanctioned_works(path: str = DEFAULT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df = df.rename(columns=COLUMN_MAP)

    # Drop the "Grand Total" footer row and any other row where amount isn't
    # a clean number (guards against future exports adding similar footers).
    amount_numeric = pd.to_numeric(
        df["amount"].str.replace(",", "", regex=False), errors="coerce"
    )
    dropped = df[amount_numeric.isna()]
    if len(dropped) > 0:
        print(f"[data_loader] Dropping {len(dropped)} row(s) with non-numeric amount "
              f"(e.g. footer/total rows): {dropped['sr_no'].tolist()}")
    df = df[amount_numeric.notna()].copy()
    df["amount"] = amount_numeric[amount_numeric.notna()].values

    df["recommended_date"] = pd.to_datetime(
        df["recommended_date"], format="%d-%b-%Y", errors="coerce"
    )
    df["sanction_date"] = pd.to_datetime(
        df["sanction_date"], format="%d-%b-%Y", errors="coerce"
    )

    bad_dates = df[df["recommended_date"].isna() | df["sanction_date"].isna()]
    if len(bad_dates) > 0:
        print(f"[data_loader] Dropping {len(bad_dates)} row(s) with unparseable dates")
    df = df[df["recommended_date"].notna() & df["sanction_date"].notna()].copy()

    df["approval_gap_days"] = (df["sanction_date"] - df["recommended_date"]).dt.days

    df["state"] = df["state"].str.strip()
    df["agency_raw"] = df["agency_raw"].str.strip()
    df["work_raw"] = df["work_raw"].str.strip()

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    d = load_sanctioned_works()
    print(f"Loaded {len(d)} works")
    print(d[["amount", "sanction_date", "recommended_date", "approval_gap_days",
              "state", "agency_raw"]].head(5).to_string(index=False))
    print(f"\napproval_gap_days: min={d['approval_gap_days'].min()}, "
          f"max={d['approval_gap_days'].max()}, "
          f"negative gaps (sanctioned before recommended — data issue): "
          f"{(d['approval_gap_days'] < 0).sum()}")