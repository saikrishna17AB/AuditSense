
# """Test one new MPLADS work with saved Module 2 artifacts.

# Run from the project root:
#     python -m FinancialAnomaly.test_single_work

# Existing project files are not modified.
# """
# import json
# from pathlib import Path

# import joblib
# import numpy as np
# import pandas as pd

# from FinancialAnomaly.features.build_features import (
#     FEATURE_COLUMNS,
#     build_raw_features,
#     add_rule_flags,
# )
# from FinancialAnomaly.shared.data_loader import load_sanctioned_works
# # from FinancialAnomaly.shared.agency_standardization import add_agency_id
# # from FinancialAnomaly.shared.work_type_classifier import add_work_type


# ROOT = Path(__file__).resolve().parent.parent
# ARTIFACT_DIR = ROOT / "Artifacts" / "module2" / "inference"


# def percentile_against_reference(value, sorted_reference):
#     """Empirical percentile; this is not a probability of fraud."""
#     ref = np.asarray(sorted_reference, dtype=float)
#     if len(ref) == 0:
#         return None
#     return round(float(100.0 * np.searchsorted(
#         ref, value, side="right"
#     ) / len(ref)), 2)


# def main():
#     required = [
#         "scaler.joblib",
#         "fusion_model.joblib",
#         "reference_scores.json",
#         "metadata.json",
#     ]
#     missing = [name for name in required if not (ARTIFACT_DIR / name).exists()]
#     if missing:
#         raise FileNotFoundError(
#             f"Missing inference artifacts in {ARTIFACT_DIR}: {missing}\n"
#             "Run: python -m FinancialAnomaly.train_for_inference"
#         )

#     scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
#     model = joblib.load(ARTIFACT_DIR / "fusion_model.joblib")

#     with open(ARTIFACT_DIR / "reference_scores.json", encoding="utf-8") as f:
#         reference = json.load(f)

#     # Load the historical dataset and derive its existing feature definitions.
#     print("Loading raw historical works...")
#     history = load_sanctioned_works()

#     print(f"Historical reference works: {len(history):,}")
#     print("\nEnter one new work. Use values appropriate to your dataset.")

#     amount = float(input("Sanctioned/recommended amount (INR): ").strip())
#     state = input("State: ").strip()
#     agency = input("Agency name: ").strip()
#     description = input("Work description (include work type): ").strip()
#     recommended_date = input("Recommended date (DD-Mon-YYYY, or blank): ").strip()
#     sanction_date = input("Sanction date (DD-Mon-YYYY, or blank): ").strip()

#     if amount <= 0:
#         raise ValueError("Amount must be greater than zero.")
#     if not state or not agency or not description:
#         raise ValueError("State, agency, and description are required.")

#     date_format = "%d-%b-%Y"

#     def parse_date(value):
#         if not value:
#             return pd.NaT
#         parsed = pd.to_datetime(value, format=date_format, errors="coerce")
#         if pd.isna(parsed):
#             raise ValueError(
#                 f"Invalid date '{value}'. Use a format such as 08-Oct-2026."
#             )
#         return parsed

#     rec_date = parse_date(recommended_date)
#     san_date = parse_date(sanction_date)

#     # Build one row with the columns expected by the existing feature functions.
#     # The input amount is used as the new work's amount.
#     new_row = {
#         "amount": amount,
#         "state": state,
#         "agency_raw": agency,
#         "work_raw": description,
#         "recommended_date": rec_date,
#         "sanction_date": san_date,
#         "approval_gap_days": (
#             (san_date - rec_date).days
#             if pd.notna(rec_date) and pd.notna(san_date)
#             else np.nan
#         ),
#     }

#     # Use historical data as context for agency statistics and baselines.
#     # Add the new row only for the per-work features and model prediction.
#     history = history.copy()
#     new_df = pd.DataFrame([new_row])

#     # Standardize the new agency/work type using the same project functions.
#     # We deliberately do not refit the saved scaler.
    
    
#     combined = pd.concat(
#         [history, new_df],
#         ignore_index=True,
#         sort=False,
#     )


#     # Recompute features with the combined table so the new work receives
#     # the same feature schema. Historical baselines may be affected slightly
#     # by including this row; this is disclosed in the output.
    
#     # Remove feature columns inherited from historical feature engineering.
#     # build_raw_features() must calculate these columns afresh.
#     derived_columns = [
#         "f1_threshold_avoidance",
#         "f2_benford_excess",
#         "f3_round_number",
#         "f4_agency_cost_trend",
#         "f5_approval_gap",
#         "f6_agency_fingerprint",
#         "flag_threshold_avoidance",
#         "flag_fast_approval",
#         "flag_pricing_outlier",
#         "year_month",
#         "n_works",
#         "n_months",
#     ]

#     combined = combined.drop(
#         columns=[c for c in derived_columns if c in combined.columns],
#         errors="ignore",
#     )

# #     scored = build_raw_features(combined)
# #     scored = add_rule_flags(scored)


# #     row = scored.iloc[[-1]].copy()
# #     raw_values = row[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

# #     if raw_values.isna().any(axis=None):
# #         missing_features = raw_values.columns[raw_values.isna().any()].tolist()
# #         raise ValueError(f"Could not calculate these features: {missing_features}")

# #     # Apply the scaler saved during training.
# #     X_scaled = scaler.transform(raw_values[FEATURE_COLUMNS])
# #     X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)

# #     # Raw model signals: higher IF anomaly value means more unusual;
# #     # higher reconstruction error means poorer reconstruction.
# #     if_raw = float(model.iso_forest.score_samples(X_scaled)[0])
# #     if_anomaly = -if_raw

# #     reconstructed = model.autoencoder.predict(X_scaled)
# #     ae_error = float(np.mean(
# #         (X_scaled.to_numpy() - reconstructed) ** 2, axis=1
# #     )[0])

# #     if_pct = percentile_against_reference(
# #         if_anomaly, reference["isolation_forest_anomaly"]
# #     )
# #     ae_pct = percentile_against_reference(
# #         ae_error, reference["autoencoder_error"]
# #     )

# #     if if_pct is None or ae_pct is None:
# #         raise ValueError("Reference score calibration data is empty.")

# #     fused_pct = round((if_pct + ae_pct) / 2.0, 2)

# #     # Evidence notes use raw features/rules, not SHAP.
# #     evidence = []
# #     feature_row = row.iloc[0]

# #     if feature_row["f1_threshold_avoidance"] > 0:
# #         evidence.append(
# #             f"Amount is within ₹50,000 below a configured scrutiny threshold "
# #             f"(score {feature_row['f1_threshold_avoidance']:.3f})."
# #         )
# #     if feature_row["f2_benford_excess"] > 0:
# #         evidence.append(
# #             f"Agency Benford deviation exceeds the dataset baseline "
# #             f"(excess score {feature_row['f2_benford_excess']:.3f})."
# #         )
# #     if feature_row["f3_round_number"] >= 0.6:
# #         evidence.append(
# #             f"Amount has a strong round-number pattern "
# #             f"(score {feature_row['f3_round_number']:.3f})."
# #         )
# #     if feature_row["f4_agency_cost_trend"] > 0:
# #         evidence.append(
# #             f"Agency monthly average cost has a rising trend "
# #             f"(standardized slope {feature_row['f4_agency_cost_trend']:.3f})."
# #         )
# #     if feature_row["f5_approval_gap"] > 0:
# #         evidence.append(
# #             f"Recommendation-to-sanction gap is unusually short "
# #             f"(score {feature_row['f5_approval_gap']:.3f})."
# #         )
# #     if feature_row["f6_agency_fingerprint"] > 3:
# #         evidence.append(
# #             f"Amount is more than 3 agency standard deviations from its "
# #             f"historical mean (z-score {feature_row['f6_agency_fingerprint']:.3f})."
# #         )

# #     print("\n" + "=" * 58)
# #     print("MODULE 2 — SINGLE WORK TEST RESULT")
# #     print("=" * 58)
# #     print(f"Amount: ₹{amount:,.2f}")
# #     print(f"State: {state}")
# #     print(f"Agency: {agency}")
# #     print(f"Description: {description}")

# #     print("\nSix raw features:")
# #     for name in FEATURE_COLUMNS:
# #         print(f"  {name}: {float(feature_row[name]):.4f}")

# #     print("\nModel signals:")
# #     print(f"  Isolation Forest raw anomaly value: {if_anomaly:.6f}")
# #     print(f"  Isolation Forest reference percentile: {if_pct:.2f}")
# #     print(f"  Autoencoder reconstruction error: {ae_error:.6f}")
# #     print(f"  Autoencoder reference percentile: {ae_pct:.2f}")
# #     print(f"  Fused reference percentile: {fused_pct:.2f}")

# #     print("\nRule flags:")
# #     print(f"  Threshold avoidance: {bool(feature_row['flag_threshold_avoidance'])}")
# #     print(f"  Fast approval: {bool(feature_row['flag_fast_approval'])}")
# #     print(f"  Pricing outlier: {bool(feature_row['flag_pricing_outlier'])}")

# #     print("\nEvidence-based explanation:")
# #     if evidence:
# #         for item in evidence:
# #             print(f"  - {item}")
# #     else:
# #         print("  - No configured feature evidence crossed the reporting thresholds.")

# #     print(
# #         "\nInterpretation: reference percentiles measure unusualness against "
# #         "historical model scores; they are NOT probabilities of fraud."
# #     )
# #     print(
# #         "Context note: this initial test recomputes feature baselines using "
# #         "the combined table. A later refinement should freeze historical "
# #         "baselines exactly for production inference."
# #     )


# # if __name__ == "__main__":
# #     main()

#     history = history.copy()
#     new_df = pd.DataFrame([new_row])

#     # Append the new work to RAW history.
#     # build_raw_features() will calculate agency IDs, work types,
#     # and all six feature columns exactly once.
#     combined = pd.concat(
#         [history, new_df],
#         ignore_index=True,
#         sort=False,
#     )

    
#     row = scored.iloc[[-1]].copy()
#     raw_values = row[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

#     if raw_values.isna().any(axis=None):
#         missing_features = raw_values.columns[
#             raw_values.isna().any()
#         ].tolist()
#         raise ValueError(
#             f"Could not calculate these features: {missing_features}"
#         )

#     # Use the scaler saved during training; do not refit it.
#     X_scaled = scaler.transform(raw_values[FEATURE_COLUMNS])

#     # Isolation Forest anomaly signal.
#     if_raw = float(model.iso_forest.score_samples(X_scaled)[0])
#     if_anomaly = -if_raw

#     # Autoencoder reconstruction error.
#     reconstructed = model.autoencoder.predict(X_scaled)
#     ae_error = float(
#         np.mean(
#             (X_scaled - reconstructed) ** 2,
#             axis=1
#         )[0]
#     )

#     # Compare with saved historical reference scores.
#     if_pct = percentile_against_reference(
#         if_anomaly,
#         reference["isolation_forest_anomaly"]
#     )
#     ae_pct = percentile_against_reference(
#         ae_error,
#         reference["autoencoder_error"]
#     )

#     if if_pct is None or ae_pct is None:
#         raise ValueError("Reference score calibration data is empty.")

#     fused_pct = round((if_pct + ae_pct) / 2.0, 2)
#     feature_row = row.iloc[0]

#     # Evidence-based explanation; this is not SHAP.
#     evidence = []

#     if feature_row["f1_threshold_avoidance"] > 0:
#         evidence.append(
#             "Amount is close to a configured scrutiny threshold."
#         )
#     if feature_row["f2_benford_excess"] > 0:
#         evidence.append(
#             "Agency Benford deviation exceeds the dataset baseline."
#         )
#     if feature_row["f3_round_number"] >= 0.6:
#         evidence.append(
#             "Amount has a strong round-number pattern."
#         )
#     if feature_row["f4_agency_cost_trend"] > 0:
#         evidence.append(
#             "Agency monthly average cost trend is positive."
#         )
#     if feature_row["f5_approval_gap"] > 0:
#         evidence.append(
#             "Recommendation-to-sanction gap is unusually short."
#         )
#     if feature_row["f6_agency_fingerprint"] > 3:
#         evidence.append(
#             "Amount is more than 3 agency standard deviations "
#             "from its historical mean."
#         )

#     print("\n" + "=" * 58)
#     print("MODULE 2 — SINGLE WORK TEST RESULT")
#     print("=" * 58)
#     print(f"Amount: ₹{amount:,.2f}")
#     print(f"State: {state}")
#     print(f"Agency: {agency}")
#     print(f"Description: {description}")

#     print("\nSix raw features:")
#     for name in FEATURE_COLUMNS:
#         print(f"  {name}: {float(feature_row[name]):.4f}")

#     print("\nModel signals:")
#     print(f"  Isolation Forest anomaly value: {if_anomaly:.6f}")
#     print(f"  Isolation Forest reference percentile: {if_pct:.2f}")
#     print(f"  Autoencoder reconstruction error: {ae_error:.6f}")
#     print(f"  Autoencoder reference percentile: {ae_pct:.2f}")
#     print(f"  Fused reference percentile: {fused_pct:.2f}")

#     print("\nRule flags:")
#     print(
#         "  Threshold avoidance:",
#         bool(feature_row["flag_threshold_avoidance"])
#     )
#     print(
#         "  Fast approval:",
#         bool(feature_row["flag_fast_approval"])
#     )
#     print(
#         "  Pricing outlier:",
#         bool(feature_row["flag_pricing_outlier"])
#     )

#     print("\nEvidence-based explanation:")
#     if evidence:
#         for item in evidence:
#             print(f"  - {item}")
#     else:
#         print("  - No configured feature evidence crossed reporting thresholds.")

#     print(
#         "\nInterpretation: reference percentiles measure unusualness "
#         "against historical model scores. They are NOT probabilities of fraud."
#     )
#     print(
#         "Note: this initial test recomputes feature baselines on the combined "
#         "historical dataset plus the new work."
#     )


# if __name__ == "__main__":
#     main()




"""Test one new MPLADS work using saved Module 2 artifacts.

Run from the project root:
    python -m FinancialAnomaly.test_single_work

This script does not modify existing project files.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from FinancialAnomaly.features.build_features import (
    FEATURE_COLUMNS,
    build_raw_features,
    add_rule_flags,
)
from FinancialAnomaly.shared.data_loader import load_sanctioned_works


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "Artifacts" / "module2" / "inference"


def percentile_against_reference(value, sorted_reference):
    """Return an empirical percentile, not a probability of fraud."""
    ref = np.asarray(sorted_reference, dtype=float)

    if len(ref) == 0:
        return None

    return round(
        float(100.0 * np.searchsorted(ref, value, side="right") / len(ref)),
        2,
    )


def main():
    # 1. Check and load the saved inference artifacts.
    required = [
        "scaler.joblib",
        "fusion_model.joblib",
        "reference_scores.json",
        "metadata.json",
    ]

    missing = [
        name for name in required
        if not (ARTIFACT_DIR / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing inference artifacts in {ARTIFACT_DIR}: {missing}\n"
            "Run: python -m FinancialAnomaly.train_for_inference"
        )

    scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
    model = joblib.load(ARTIFACT_DIR / "fusion_model.joblib")

    with open(
        ARTIFACT_DIR / "reference_scores.json",
        encoding="utf-8",
    ) as f:
        reference = json.load(f)

    # 2. Load RAW historical works.
    print("Loading raw historical works...")
    history = load_sanctioned_works()

    print(f"Historical reference works: {len(history):,}")
    print("\nEnter one new work.")

    # 3. Collect the new work's details.
    amount = float(input("Sanctioned/recommended amount (INR): ").strip())
    state = input("State: ").strip()
    agency = input("Agency name: ").strip()
    description = input("Work description (include work type): ").strip()
    recommended_date = input(
        "Recommended date (DD-Mon-YYYY, or blank): "
    ).strip()
    sanction_date = input(
        "Sanction date (DD-Mon-YYYY, or blank): "
    ).strip()

    if not np.isfinite(amount) or amount <= 0:
        raise ValueError("Amount must be a finite number greater than zero.")

    if not state or not agency or not description:
        raise ValueError("State, agency, and description are required.")

    date_format = "%d-%b-%Y"

    def parse_date(value):
        if not value:
            return pd.NaT

        parsed = pd.to_datetime(
            value,
            format=date_format,
            errors="coerce",
        )

        if pd.isna(parsed):
            raise ValueError(
                f"Invalid date '{value}'. Use a format such as 08-Oct-2026."
            )

        return parsed

    rec_date = parse_date(recommended_date)
    san_date = parse_date(sanction_date)

    if pd.notna(rec_date) and pd.notna(san_date) and san_date < rec_date:
        raise ValueError(
            "Sanction date cannot be earlier than recommendation date."
        )

    new_row = {
        "amount": amount,
        "state": state,
        "agency_raw": agency,
        "work_raw": description,
        "recommended_date": rec_date,
        "sanction_date": san_date,
        "approval_gap_days": (
            (san_date - rec_date).days
            if pd.notna(rec_date) and pd.notna(san_date)
            else np.nan
        ),
    }

    # 4. Append the new row to RAW history.
    # Do not precompute features or add agency/work-type IDs here.
    combined = pd.concat(
        [history.copy(), pd.DataFrame([new_row])],
        ignore_index=True,
        sort=False,
    )

    # 5. Calculate features exactly once.
    print("\nCalculating features for the combined dataset...")
    scored = build_raw_features(combined)
    scored = add_rule_flags(scored)

    # 6. Select only the new work.
    row = scored.iloc[[-1]].copy()
    feature_row = row.iloc[0]

    raw_values = row[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if raw_values.isna().any(axis=None):
        missing_features = raw_values.columns[
            raw_values.isna().any()
        ].tolist()

        raise ValueError(
            f"Could not calculate these features: {missing_features}"
        )

    
    feature_names = list(FEATURE_COLUMNS)

    X_scaled = scaler.transform(raw_values[feature_names])
    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=feature_names,
    )

    if_raw = float(model.iso_forest.score_samples(X_scaled_df)[0])
    if_anomaly = -if_raw

    reconstructed = model.autoencoder.predict(X_scaled_df)
    ae_error = float(
        np.mean(
            (X_scaled_df.to_numpy() - reconstructed) ** 2,
            axis=1,
        )[0]
    )


    # 10. Compare scores against saved reference distributions.
    if_pct = percentile_against_reference(
        if_anomaly,
        reference["isolation_forest_anomaly"],
    )

    ae_pct = percentile_against_reference(
        ae_error,
        reference["autoencoder_error"],
    )

    if if_pct is None or ae_pct is None:
        raise ValueError("Reference score calibration data is empty.")

    # This is an average of two empirical percentiles, not a fraud probability.
    fused_pct = round((if_pct + ae_pct) / 2.0, 2)

    # 11. Create evidence-based explanations from raw features.
    evidence = []

    if feature_row["f1_threshold_avoidance"] > 0:
        evidence.append(
            "Amount is close to a configured scrutiny threshold "
            f"(score {feature_row['f1_threshold_avoidance']:.3f})."
        )

    if feature_row["f2_benford_excess"] > 0:
        evidence.append(
            "Agency Benford deviation exceeds the dataset baseline "
            f"(score {feature_row['f2_benford_excess']:.3f})."
        )

    if feature_row["f3_round_number"] >= 0.6:
        evidence.append(
            "Amount has a strong round-number pattern "
            f"(score {feature_row['f3_round_number']:.3f})."
        )

    if feature_row["f4_agency_cost_trend"] > 0:
        evidence.append(
            "Agency monthly average cost trend is positive "
            f"(score {feature_row['f4_agency_cost_trend']:.3f})."
        )

    if feature_row["f5_approval_gap"] > 0:
        evidence.append(
            "Recommendation-to-sanction gap is unusually short "
            f"(score {feature_row['f5_approval_gap']:.3f})."
        )

    if feature_row["f6_agency_fingerprint"] > 3:
        evidence.append(
            "Amount is more than 3 agency standard deviations "
            "from its historical mean "
            f"(z-score {feature_row['f6_agency_fingerprint']:.3f})."
        )

    # 12. Print results.
    print("\n" + "=" * 58)
    print("MODULE 2 — SINGLE WORK TEST RESULT")
    print("=" * 58)

    print(f"Amount: ₹{amount:,.2f}")
    print(f"State: {state}")
    print(f"Agency: {agency}")
    print(f"Description: {description}")

    print("\nSix raw features:")
    for name in FEATURE_COLUMNS:
        print(f"  {name}: {float(feature_row[name]):.4f}")

    print("\nModel signals:")
    print(f"  Isolation Forest anomaly value: {if_anomaly:.6f}")
    print(f"  Isolation Forest reference percentile: {if_pct:.2f}")
    print(f"  Autoencoder reconstruction error: {ae_error:.6f}")
    print(f"  Autoencoder reference percentile: {ae_pct:.2f}")
    print(f"  Fused reference percentile: {fused_pct:.2f}")

    print("\nRule flags:")
    print(
        "  Threshold avoidance:",
        bool(feature_row["flag_threshold_avoidance"]),
    )
    print(
        "  Fast approval:",
        bool(feature_row["flag_fast_approval"]),
    )
    print(
        "  Pricing outlier:",
        bool(feature_row["flag_pricing_outlier"]),
    )

    print("\nEvidence-based explanation (not SHAP):")
    if evidence:
        for item in evidence:
            print(f"  - {item}")
    else:
        print(
            "  - No configured feature evidence crossed the "
            "reporting thresholds."
        )

    print(
        "\nInterpretation: reference percentiles measure unusualness "
        "against historical model scores. They are NOT probabilities of fraud."
    )

    print(
        "Baseline note: this test calculates features on the combined "
        "historical dataset and new work. Some dataset-level and agency-level "
        "features may therefore be influenced by the new row."
    )


if __name__ == "__main__":
    main()
