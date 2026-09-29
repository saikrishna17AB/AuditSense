"""f5 - Approval-gap score.

Unusually fast recommendation->sanction gaps can mean bypassed scrutiny.
Unusually slow gaps are a separate signal (delay), not this feature's job.

Score = 1.0 for same-day/near-instant sanctions, decaying to 0 by GAP_CAP days.
"""
import pandas as pd

from FinancialAnomaly.shared.data_loader import load_sanctioned_works
from FinancialAnomaly.shared.agency_standardization import add_agency_id

GAP_CAP = 7  # gaps >= 7 days score 0; below that, score rises toward 1 at 0 days


def _gap_score(days: int) -> float:
    if days is None or days < 0:
        return 0.0
    if days >= GAP_CAP:
        return 0.0
    return round(1.0 - (days / GAP_CAP), 4)


def compute_approval_gap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["f5_approval_gap"] = df["approval_gap_days"].apply(_gap_score)
    return df


if __name__ == "__main__":
    d = load_sanctioned_works()
    d = add_agency_id(d)
    d = compute_approval_gap(d)

    print("approval_gap_days distribution:")
    print(d["approval_gap_days"].describe())

    flagged = d[d["f5_approval_gap"] > 0].sort_values("f5_approval_gap", ascending=False)
    print(f"\nWorks sanctioned within {GAP_CAP} days of recommendation: {len(flagged)} of {len(d)}")
    print(flagged[["agency_raw", "amount", "approval_gap_days", "f5_approval_gap"]].head(10).to_string(index=False))

    over_75 = (d["approval_gap_days"] > 75).sum()
    print(f"\nWorks with gap > 75 days: {over_75} of {len(d)} ({100*over_75/len(d):.1f}%)")