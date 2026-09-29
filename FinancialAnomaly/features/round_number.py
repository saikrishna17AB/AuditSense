"""f3 - Round-number score.

Genuine itemized costs rarely land on perfectly round figures. An agency
with an unusually high rate of round amounts (₹5,00,000, ₹4,50,000, etc.)
compared to the overall population is a signal worth flagging — separate
from threshold-avoidance (f1), which only cares about proximity to a
specific limit, not roundness in general.

Per-work score (0-1): how "round" a single amount is, based on trailing zeros.
Per-agency flag: agencies whose round-rate is well above the dataset average.
"""
import pandas as pd

from FinancialAnomaly.shared.data_loader import load_sanctioned_works
from FinancialAnomaly.shared.agency_standardization import add_agency_id
from FinancialAnomaly.shared.work_type_classifier import add_work_type


def _round_score(amount: float) -> float:
    amount = int(amount)
    if amount == 0:
        return 0.0
    if amount % 100_000 == 0:
        return 1.0
    if amount % 50_000 == 0:
        return 0.8
    if amount % 10_000 == 0:
        return 0.6
    if amount % 1_000 == 0:
        return 0.3
    if amount % 100 == 0:
        return 0.1
    return 0.0


def compute_round_number(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["f3_round_number"] = df["amount"].apply(_round_score)
    return df


def category_round_rate(df: pd.DataFrame, min_works: int = 20) -> pd.DataFrame:
    """Baseline round-rate PER WORK TYPE (derived from description text, see
    work_type_classifier.py). Some types (e.g. standardized RO systems, street
    lights) are legitimately round by policy — comparing an agency against the
    global rate would misflag every agency doing that work.
    """
    is_round = (df["f3_round_number"] >= 0.6).astype(int)
    grp = df.assign(is_round=is_round).groupby("work_type").agg(
        n_works=("amount", "size"),
        round_rate=("is_round", "mean"),
    )
    return grp[grp["n_works"] >= min_works].sort_values("round_rate", ascending=False)


def agency_round_rate_vs_category(df: pd.DataFrame, min_works: int = 5) -> pd.DataFrame:
    """Per-agency round-rate, compared against the baseline for the SAME
    (state, work_type) pair where there's enough data for that pair;
    falls back to work_type-only baseline when the state-specific slice is
    too small (state costs likely vary, but sample size matters more).
    """
    is_round = (df["f3_round_number"] >= 0.6).astype(int)
    df = df.assign(is_round=is_round)

    # baseline 1: state x work_type (preferred, more specific)
    state_type_baseline = df.groupby(["state", "work_type"])["is_round"].agg(["mean", "count"])
    state_type_baseline = state_type_baseline.rename(columns={"mean": "baseline_rate", "count": "baseline_n"})

    # baseline 2: work_type only (fallback)
    type_baseline = df.groupby("work_type")["is_round"].mean().rename("type_fallback_rate")

    agency_grp = df.groupby(["agency_id", "state", "work_type"]).agg(
        n_works=("amount", "size"),
        round_rate=("is_round", "mean"),
    ).reset_index()
    agency_grp = agency_grp[agency_grp["n_works"] >= min_works]

    agency_grp = agency_grp.merge(state_type_baseline, on=["state", "work_type"], how="left")
    agency_grp = agency_grp.merge(type_baseline, on="work_type", how="left")

    MIN_BASELINE_N = 15  # need at least this many (state,work_type) samples to trust that baseline
    agency_grp["baseline_used"] = agency_grp["baseline_rate"].where(
        agency_grp["baseline_n"] >= MIN_BASELINE_N, agency_grp["type_fallback_rate"]
    )
    agency_grp["vs_baseline"] = agency_grp["round_rate"] - agency_grp["baseline_used"]

    return agency_grp.sort_values("vs_baseline", ascending=False)


if __name__ == "__main__":
    d = load_sanctioned_works()
    d = add_agency_id(d)
    d = add_work_type(d)
    d = compute_round_number(d)

    print(f"Works with round score >= 0.6: {(d['f3_round_number'] >= 0.6).sum()} of {len(d)}")

    cat_rates = category_round_rate(d)
    print(f"\nRound-rate BY WORK TYPE (min 20 works) — this is the real baseline:")
    print(cat_rates.to_string())

    print(f"\nAgencies rounder than their OWN (state, work_type) baseline, or work_type")
    print(f"fallback if state-specific sample too small (min 5 works/group):")
    ranked = agency_round_rate_vs_category(d)
    print(ranked[ranked["vs_baseline"] > 0.3].head(10).to_string(index=False))