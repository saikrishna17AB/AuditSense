"""f6 - Agency fingerprint score (per-work z-score vs. that agency's own baseline).

Different from f4 (agency_cost_trend), which asks "is this agency's average
cost rising/falling over time." f6 asks a simpler, per-work question: "is
THIS ONE amount unusual compared to what this agency normally sanctions,"
regardless of when. Catches a single outlier bill from an otherwise
consistent agency, which f4's trend-line wouldn't necessarily flag.

Agencies with too few works to get a meaningful mean/std get score 0
(not flagged) rather than a noisy/unstable z-score from 2-3 data points.
"""
import pandas as pd

from FinancialAnomaly.shared.data_loader import load_sanctioned_works
from FinancialAnomaly.shared.agency_standardization import add_agency_id

MIN_WORKS_FOR_BASELINE = 5


def compute_agency_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    stats = df.groupby("agency_id")["amount"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "agency_mean", "std": "agency_std", "count": "agency_n"}
    )
    df = df.merge(stats, on="agency_id", how="left")

    def _z(row):
        if row["agency_n"] < MIN_WORKS_FOR_BASELINE or row["agency_std"] in (0, None) or pd.isna(row["agency_std"]):
            return 0.0
        return round(abs(row["amount"] - row["agency_mean"]) / row["agency_std"], 4)

    df["f6_agency_fingerprint"] = df.apply(_z, axis=1)
    return df.drop(columns=["agency_mean", "agency_std", "agency_n"])


if __name__ == "__main__":
    d = load_sanctioned_works()
    d = add_agency_id(d)
    d = compute_agency_fingerprint(d)

    scored = d[d["f6_agency_fingerprint"] > 0]
    print(f"Works with a valid agency baseline (agency has >= {MIN_WORKS_FOR_BASELINE} works): "
          f"{len(scored)} of {len(d)}")

    print(f"\nTop 10 works most unusual vs. their OWN agency's normal amount (highest z-score):")
    top = scored.sort_values("f6_agency_fingerprint", ascending=False).head(10)
    print(top[["agency_raw", "amount", "f6_agency_fingerprint"]].to_string(index=False))

    print(f"\nWorks with z-score > 3 (statistically extreme even for their own agency): "
          f"{(scored['f6_agency_fingerprint'] > 3).sum()}")