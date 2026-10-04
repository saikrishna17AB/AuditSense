"""Builds the feature vector for Time_Cost_Overrun. Self-contained - does
NOT depend on FinancialAnomaly's internals, so this module has no
cross-package import risk.

Agency track record combines TWO sources:
  1. prior_reliability_rate - Datasets/MPLADS.csv (FY23-24), genuinely
     leakage-safe (see features/prior_history.py).
  2. agency_alltime_*_rate  - each agency's own rate across ALL its works
     in the CURRENT (2024-2026) dataset. NOT strictly leakage-safe (a
     disclosed simplification) - the strict leakage-safe version
     (production_leakage_safe_rate below) is ~99% cold-start in this
     ~24-month window, since completions routinely take longer than the
     gap between an agency's own sanction dates.
approval_gap_days (recommendation -> sanction gap) is included too.
"""
import re
import numpy as np
import pandas as pd

from Time_Cost_Overrun.data_loader import load_ground_truth
from Time_Cost_Overrun.features.prior_history import add_prior_reliability

FEATURES_NUM = ["sanctioned_amount", "approval_gap_days",
                "prior_reliability_rate", "agency_alltime_time_rate", "agency_alltime_underutil_rate"]
FEATURES_CAT = ["state", "work_type", "sanction_season"]

_SEASON_MAP = {12: "winter", 1: "winter", 2: "winter",
               3: "summer", 4: "summer", 5: "summer",
               6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
               10: "post_monsoon", 11: "post_monsoon"}

_WORK_TYPE_PREFIX_RE = re.compile(r"^WS/\s*MP\d+/\d{4}-\d{4}/\d+-")
_AGENCY_ROLE_RULES = [
    (r"PLANNING", "DISTRICT PLANNING OFFICER"),
    (r"MAGI", "DISTRICT MAGISTRATE"),
    (r"COLLECT", "DISTRICT COLLECTOR"),
    (r"COMM?I|\bDC\b", "DEPUTY COMMISSIONER"),
]


def _agency_id(raw) -> str:
    if not isinstance(raw, str):
        return "UNKNOWN"
    name = re.sub(r"\s+", " ", raw.upper().strip())
    m = re.match(r"^([A-Z0-9 &]+)\(", name)
    district = m.group(1).strip() if m else name
    inner = name[name.find("("):].replace("_", " ") if "(" in name else name
    role = next((canon for pat, canon in _AGENCY_ROLE_RULES if re.search(pat, inner)), "UNKNOWN_ROLE")
    return f"{district} | {role}"


def _work_type(raw) -> str:
    if not isinstance(raw, str):
        return "UNKNOWN"
    cleaned = _WORK_TYPE_PREFIX_RE.sub("", raw).strip()
    return cleaned if cleaned else "UNKNOWN"


def production_leakage_safe_rate(df: pd.DataFrame, label_col: str, out_col: str) -> pd.Series:
    """PRODUCTION-INTENDED version: each row gets its agency's rate among
    works completed strictly before this row's own sanction_date. Not
    used in FEATURES_NUM in this ~24-month window (~99% cold-start) -
    kept here for a future deployment with longer data history."""
    d = df[["agency_id", "sanction_date", "completion_date", label_col]].copy()
    result = np.empty(len(d))
    for agency_id, idx in d.groupby("agency_id").groups.items():
        sub = d.loc[idx].sort_values("completion_date")
        dates = sub["completion_date"].to_numpy()
        labels = sub[label_col].to_numpy()
        cum_n = np.arange(1, len(sub) + 1)
        cum_sum = labels.cumsum()
        for i, row in sub.iterrows():
            pos = np.searchsorted(dates, row["sanction_date"], side="left")
            result[d.index.get_loc(i)] = (cum_sum[pos - 1] / cum_n[pos - 1]) if pos > 0 else np.nan
    return pd.Series(result, index=d.index, name=out_col)


def _alltime_rate(df: pd.DataFrame, label_col: str) -> pd.Series:
    return df.groupby("agency_id")[label_col].transform("mean")


def build_features() -> pd.DataFrame:
    df = load_ground_truth()
    df["agency_id"] = df["agency_raw"].apply(_agency_id)
    df["work_type"] = df["work_raw"].apply(_work_type)
    df = add_prior_reliability(df, raw_col="agency_raw")

    df["approval_gap_days"] = (df["sanction_date"] - df["recommended_date"]).dt.days
    df["sanction_season"] = df["sanction_date"].dt.month.map(_SEASON_MAP)
    df["time_overrun_label"] = (df["execution_days"] > 365).astype(int)
    df["underutilization_label"] = (df["under_utilization_pct"] > 0.01).astype(int)

    df["agency_alltime_time_rate"] = _alltime_rate(df, "time_overrun_label")
    df["agency_alltime_underutil_rate"] = _alltime_rate(df, "underutilization_label")
    return df


if __name__ == "__main__":
    d = build_features()
    print("Rows:", len(d))
    print(d[FEATURES_NUM].describe().T[["mean", "std", "min", "max"]])
    print("\ntime_overrun_label rate:", round(d["time_overrun_label"].mean(), 3))
    print("underutilization_label rate:", round(d["underutilization_label"].mean(), 3))