"""Assembles the six per-work features into one matrix, ready for
Isolation Forest / Autoencoder, plus the rule-based flags that run
independently of the models.

Feature scaling: f1, f3, f5 are already 0-1 by construction; f6 is a raw,
unbounded z-score (seen up to ~9 in this dataset). Feeding that mix
straight into a model lets f6 dominate every split/reconstruction, so
every feature is standardized (z-score) here before stacking, per the
original preprocessing plan.

Rule-based flags (separate from the model, self-explanatory on their own):
  - flag_threshold_avoidance: amount sits right up against a scrutiny limit
  - flag_fast_approval: sanctioned same-day or next-day
  - flag_pricing_outlier: added after the injection-test run below surfaced
    a real gap — an isolated pricing outlier with no other red flag was
    barely caught by the model (recall 2.2%), because f6 gets compressed
    once standardized alongside four 0-1-range features. This rule flag
    surfaces it directly off the RAW z-score, before scaling.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

from FinancialAnomaly.shared.data_loader import load_sanctioned_works
from FinancialAnomaly.shared.agency_standardization import add_agency_id
from FinancialAnomaly.shared.work_type_classifier import add_work_type

from FinancialAnomaly.features.threshold_avoidance import compute_threshold_avoidance
from FinancialAnomaly.features.round_number import compute_round_number
from FinancialAnomaly.features.agency_cost_trend import compute_agency_cost_trend
from FinancialAnomaly.features.approval_gap import compute_approval_gap
from FinancialAnomaly.features.agency_fingerprint import compute_agency_fingerprint
from FinancialAnomaly.features.benford import benford_chi_square, agency_benford_flags

FEATURE_COLUMNS = [
    "f1_threshold_avoidance",
    "f2_benford_excess",
    "f3_round_number",
    "f4_agency_cost_trend",
    "f5_approval_gap",
    "f6_agency_fingerprint",
]

PRICING_OUTLIER_Z = 3.0  # raw f6 z-score cutoff for the rule flag


def build_raw_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Loads (if not given) and computes all six raw, unscaled features
    per work. Returns the full dataframe — feature columns plus everything
    needed for reporting (agency_raw, state, work_type, amount, dates)."""
    if df is None:
        df = load_sanctioned_works()
    df = add_agency_id(df)
    df = add_work_type(df)

    df = compute_threshold_avoidance(df)
    df = compute_round_number(df)
    df = compute_agency_cost_trend(df)
    df = compute_approval_gap(df)
    df = compute_agency_fingerprint(df)

    # Benford is agency-level; merge its excess score back onto each work.
    # Agencies below MIN_WORKS_FOR_BENFORD get 0 (not flagged), same pattern
    # used everywhere else for agencies with too little data to score.
    _, _, dataset_v = benford_chi_square(df["amount"])
    benford_flags = agency_benford_flags(df, baseline_v=dataset_v)
    df = df.merge(
        benford_flags[["agency_id", "f2_benford_excess"]], on="agency_id", how="left"
    )
    df["f2_benford_excess"] = df["f2_benford_excess"].fillna(0.0)

    return df


def add_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flag_threshold_avoidance"] = df["f1_threshold_avoidance"] > 0
    df["flag_fast_approval"] = df["f5_approval_gap"] >= 1.0
    df["flag_pricing_outlier"] = df["f6_agency_fingerprint"] > PRICING_OUTLIER_Z
    return df


def build_features(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (full_df, scaled_feature_matrix).
    full_df keeps every raw feature, rule flags, and reporting columns.
    scaled_feature_matrix is the z-scored f1-f6 matrix the models consume.
    """
    df = build_raw_features(df)
    df = add_rule_flags(df)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[FEATURE_COLUMNS])
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=df.index)

    return df, scaled_df


if __name__ == "__main__":
    full_df, scaled_df = build_features()

    print(f"Built features for {len(full_df)} works")
    print(f"\nRaw feature ranges (before scaling):")
    print(full_df[FEATURE_COLUMNS].describe().loc[["min", "mean", "max"]].to_string())

    print(f"\nScaled feature ranges (fed to model):")
    print(scaled_df.describe().loc[["min", "mean", "max"]].to_string())

    print(f"\nRule flags:")
    print(f"  flag_threshold_avoidance: {full_df['flag_threshold_avoidance'].sum()}")
    print(f"  flag_fast_approval: {full_df['flag_fast_approval'].sum()}")
    print(f"  flag_pricing_outlier: {full_df['flag_pricing_outlier'].sum()}")

    all_three = (
        full_df["flag_threshold_avoidance"]
        & full_df["flag_fast_approval"]
    ).sum()
    print(f"  both threshold + fast approval: {all_three}")