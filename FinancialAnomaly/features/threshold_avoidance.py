"""f1 - Threshold-avoidance score.

Flags sanctioned amounts that sit suspiciously close, from below, to a known
MPLADS scrutiny threshold (₹25,00,000 and ₹15,00,000) — a pattern known as
"structuring": keeping amounts just under a limit that would otherwise
trigger extra approval/audit requirements.

Score = 1.0 means the amount sits right at the threshold (from below).
Score = 0.0 means the amount is outside the proximity window entirely.
Only amounts below a threshold, within WINDOW rupees of it, get a score;
everything else scores 0 for that threshold.
"""
import pandas as pd

from FinancialAnomaly.shared.data_loader import load_sanctioned_works
from FinancialAnomaly.shared.agency_standardization import add_agency_id

THRESHOLDS = [2_500_000, 1_500_000]   # ₹25L, ₹15L
WINDOW = 50_000                       # only count amounts within ₹50,000 below a threshold


def _threshold_score(amount: float) -> float:
    best = 0.0
    for t in THRESHOLDS:
        if 0 < t - amount <= WINDOW:
            closeness = 1.0 - (t - amount) / WINDOW
            best = max(best, closeness)
    return round(best, 4)


def compute_threshold_avoidance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["f1_threshold_avoidance"] = df["amount"].apply(_threshold_score)
    return df


if __name__ == "__main__":
    d = load_sanctioned_works()
    d = add_agency_id(d)
    d = compute_threshold_avoidance(d)

    flagged = d[d["f1_threshold_avoidance"] > 0].sort_values(
        "f1_threshold_avoidance", ascending=False
    )
    print(f"\nWorks within ₹{WINDOW:,} below a threshold: {len(flagged)} of {len(d)}")
    print(flagged[["agency_raw", "amount", "f1_threshold_avoidance"]].head(10).to_string(index=False))

    print(f"\n=== State-level aggregation (thresholds are fixed national policy limits,")
    print(f"so this doesn't change the score — it shows WHERE the pattern concentrates) ===")
    state_summary = d.groupby("state").agg(
        n_works=("amount", "size"),
        n_flagged=("f1_threshold_avoidance", lambda s: (s > 0).sum()),
    )
    state_summary["flag_rate"] = (state_summary["n_flagged"] / state_summary["n_works"]).round(4)
    state_summary = state_summary[state_summary["n_works"] >= 50].sort_values("flag_rate", ascending=False)
    print(state_summary.head(10).to_string())