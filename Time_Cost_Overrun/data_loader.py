"""Joins Datasets/Works_Sanctioned.csv with Datasets/Works Completed.csv
on a shared work ID from the 'Work' column
(WS/MP<n>/<yrs>/<serial>-<description>). 13,119 works join successfully
after dropping rows with missing amount/date or completion-before-sanction.

cost_overrun_pct is NOT produced: disbursed_amount never exceeds
sanctioned_amount in this data (0 of 13,119 works) - MPLADS Guidelines
Para 4.16/5.3 require unspent savings to be refunded. See
synthetic_cost_overrun/ for the simulated demo of that scenario instead.
"""
import re
import pandas as pd

DATASETS_DIR = "Datasets"
SANCTIONED_FILE = "Works_Sanctioned.csv"
COMPLETED_FILE = "Works Completed.csv"

_KEY_RE = re.compile(r"/(\d{5,})-")


def _extract_key(s):
    if not isinstance(s, str):
        return None
    s = re.sub(r"\s+", "", s)
    m = _KEY_RE.search(s)
    return m.group(1) if m else None


def load_ground_truth(datasets_dir: str = DATASETS_DIR) -> pd.DataFrame:
    sanc = pd.read_csv(f"{datasets_dir}/{SANCTIONED_FILE}")
    comp = pd.read_csv(f"{datasets_dir}/{COMPLETED_FILE}")

    sanc["work_key"] = sanc["Work"].apply(_extract_key)
    comp["work_key"] = comp["Work"].apply(_extract_key)

    sanc_amt = sanc["Sanction Amount ( ₹ )"].astype(str).str.replace(",", "", regex=False).str.strip()
    sanc["sanctioned_amount"] = pd.to_numeric(sanc_amt, errors="coerce")
    sanc["sanction_date"] = pd.to_datetime(sanc["Sanction Date"], format="%d-%b-%Y", errors="coerce")
    sanc["recommended_date"] = pd.to_datetime(sanc["Recommended date"], format="%d-%b-%Y", errors="coerce")

    comp_amt = comp["Amount Disbursed ( ₹ )"].astype(str).str.replace(",", "", regex=False).str.strip()
    comp["disbursed_amount"] = pd.to_numeric(comp_amt, errors="coerce")
    comp["completion_date"] = pd.to_datetime(comp["Completion Date"], format="%d-%b-%Y", errors="coerce")

    sanc = sanc.rename(columns={
        "Work": "work_raw", "IDA": "agency_raw", "Work category": "work_category",
        "State": "state", "Hon'ble Members of Parliament": "mp_name", "Constituency": "constituency",
    })[["work_key", "work_raw", "agency_raw", "work_category", "state", "mp_name", "constituency",
        "sanctioned_amount", "sanction_date", "recommended_date"]]

    comp = comp[["work_key", "disbursed_amount", "completion_date"]]
    df = sanc.merge(comp, on="work_key", how="inner", validate="one_to_one")

    before = len(df)
    df = df.dropna(subset=["sanctioned_amount", "sanction_date", "disbursed_amount", "completion_date"])
    if before - len(df):
        print(f"[data_loader] Dropped {before - len(df)} row(s) with missing amount/date")

    df["execution_days"] = (df["completion_date"] - df["sanction_date"]).dt.days
    df["under_utilization_pct"] = (df["sanctioned_amount"] - df["disbursed_amount"]) / df["sanctioned_amount"]

    before = len(df)
    df = df[df["execution_days"] >= 0]
    if before - len(df):
        print(f"[data_loader] Dropped {before - len(df)} row(s) with completion before sanction")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    d = load_ground_truth()
    print("Joined works:", len(d))
    print("Works > 365 days:", int((d["execution_days"] > 365).sum()))
    print("Works > 1% under-utilized:", int((d["under_utilization_pct"] > 0.01).sum()))