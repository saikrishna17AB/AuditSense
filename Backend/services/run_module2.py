
from pathlib import Path
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from FinancialAnomaly.features.build_features import (
    build_features,
    FEATURE_COLUMNS,
)
from FinancialAnomaly.model.fusion_model import (
    FusionModel,
    top_contributing_factors,
)


def main():
    output_dir = PROJECT_ROOT / "Artifacts" / "module2"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running Module 2 financial anomaly detection...")
    print("Building financial features from sanctioned works...")

    # Build raw features and the standardized matrix used by the models.
    full_df, scaled_df = build_features()

    if full_df.empty:
        raise RuntimeError("Module 2 produced no work records.")

    if scaled_df.isna().any().any():
        raise ValueError(
            "The scaled feature matrix contains missing values. "
            "Inspect the feature-building functions before scoring."
        )

    print(f"Works prepared: {len(full_df):,}")
    print("Training Isolation Forest and autoencoder...")

    model = FusionModel().fit(scaled_df)

    # Historical batch scoring using the existing model implementation.
    scores = model.score(scaled_df)
    result = full_df.join(scores)

    # Store readable feature evidence for each work.
    result["top_contributing_factors"] = result.apply(
        lambda row: "; ".join(
            top_contributing_factors(row, FEATURE_COLUMNS, top_n=3)
        ),
        axis=1,
    )

    # Preserve the separate rule flags and model agreement tier.
    output_file = output_dir / "module2_scores.csv"
    result.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\nModule 2 run complete")
    print("---------------------")
    print(f"Works scored: {len(result):,}")
    print(f"Output file: {output_file}")

    score_columns = [
        "isolation_forest_score",
        "autoencoder_score",
        "fused_score",
    ]
    print("\nScore summaries:")
    print(result[score_columns].describe().round(2).to_string())

    print("\nModel agreement tiers:")
    print(result["agreement_tier"].value_counts(dropna=False).to_string())

    rule_columns = [
        "flag_threshold_avoidance",
        "flag_fast_approval",
        "flag_pricing_outlier",
    ]
    print("\nIndependent rule flags:")
    for column in rule_columns:
        if column in result.columns:
            print(f"{column}: {int(result[column].sum()):,}")

    print("\nTop 10 works by fused ranking score:")
    display_columns = [
        column for column in [
            "sr_no",
            "agency_id",
            "amount",
            "work_type",
            "isolation_forest_score",
            "autoencoder_score",
            "fused_score",
            "agreement_tier",
            "top_contributing_factors",
        ]
        if column in result.columns
    ]
    print(
        result.sort_values("fused_score", ascending=False)[display_columns]
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nNote: fused_score is a ranking aid. The model's agreement_tier "
        "preserves whether the two detectors agree."
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=Warning)
        main()
