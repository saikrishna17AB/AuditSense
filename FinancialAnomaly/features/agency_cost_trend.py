"""f4 - Agency cost-trend score.

NOTE ON DATA REALITY: MPLADS eSAKSHI only went live April 2023, so
Works_Sanctioned.csv only spans 2024-2026 (~24 months), almost entirely
2024-2025. True year-over-year trend isn't meaningful with this little
history. This feature works at MONTH granularity instead, and only scores
agencies with enough spread across months to fit a real trend line —
agencies with too few distinct months get score 0 (not flagged, not
penalized), rather than forcing a trend out of insufficient data.

Score = how much an agency's monthly average cost is rising, standardized
by that agency's own cost variability (so a big agency with naturally
large swings isn't flagged the same as a small agency with a real,
consistent climb).
"""
import numpy as np
import pandas as pd

from FinancialAnomaly.shared.data_loader import load_sanctioned_works
from FinancialAnomaly.shared.agency_standardization import add_agency_id

MIN_DISTINCT_MONTHS = 4   # need at least this many distinct months to fit a trend
MIN_WORKS = 8             # and at least this many works total


def _agency_trend_slope(sub: pd.DataFrame) -> tuple[float, int]:
    """Returns (standardized_slope, n_months) for one agency's works."""
    monthly = sub.groupby("year_month")["amount"].mean().sort_index()
    n_months = len(monthly)
    if n_months < MIN_DISTINCT_MONTHS or len(sub) < MIN_WORKS:
        return 0.0, n_months

    x = np.arange(n_months)
    y = monthly.values
    slope, _ = np.polyfit(x, y, 1)

    std = sub["amount"].std()
    if std == 0 or np.isnan(std):
        return 0.0, n_months

    standardized_slope = slope / std  # slope in "own std-devs per month" units
    return round(float(standardized_slope), 4), n_months


def compute_agency_cost_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_month"] = df["sanction_date"].dt.to_period("M")

    results = []
    for agency_id, sub in df.groupby("agency_id"):
        slope, n_months = _agency_trend_slope(sub)
        results.append({"agency_id": agency_id, "n_works": len(sub),
                         "n_months": n_months, "f4_agency_cost_trend": slope})

    trend_df = pd.DataFrame(results)
    return df.merge(trend_df, on="agency_id", how="left")


if __name__ == "__main__":
    d = load_sanctioned_works()
    d = add_agency_id(d)
    d = compute_agency_cost_trend(d)

    scored = d[["agency_id", "n_works", "n_months", "f4_agency_cost_trend"]].drop_duplicates()
    print(f"Agencies with enough data to score (>= {MIN_DISTINCT_MONTHS} months, >= {MIN_WORKS} works): "
          f"{(scored['f4_agency_cost_trend'] != 0).sum()} of {len(scored)}")

    print("\nTop 10 agencies by RISING cost trend:")
    print(scored.sort_values("f4_agency_cost_trend", ascending=False).head(10).to_string(index=False))

    print("\nTop 10 agencies by FALLING cost trend:")
    print(scored.sort_values("f4_agency_cost_trend", ascending=True).head(10).to_string(index=False))