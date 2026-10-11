
"""Standalone single-work test for Module 1.
Does not modify the dashboard or overwrite existing prediction CSVs.
"""

import re
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from Time_Cost_Overrun.leakage_safe.build_features import (
    build_features,
    FEATURES_NUM,
    FEATURES_CAT,
    _agency_id,
    _work_type,
    _SEASON_MAP,
)
from Time_Cost_Overrun.features.prior_history import add_prior_reliability

LOW, MED = 40, 70


def risk_tier(score):
    if score < LOW:
        return "Low"
    if score < MED:
        return "Medium"
    return "High"


def parse_date(prompt):
    value = input(prompt).strip()
    if not value:
        return pd.NaT
    return pd.to_datetime(value, dayfirst=True, errors="raise")


def main():
    print("\nMODULE 1 — SINGLE-WORK TEST")
    print("Enter dates like 15-Jan-2024.\n")

    amount = float(input("Sanctioned amount (INR): ").replace(",", ""))
    state = input("State: ").strip()
    agency = input("Agency name: ").strip()
    description = input("Work description / type: ").strip()
    recommended_date = parse_date("Recommended date: ")
    sanction_date = parse_date("Sanction date: ")

    if amount <= 0:
        raise ValueError("Sanctioned amount must be greater than zero.")
    if pd.isna(sanction_date) or pd.isna(recommended_date):
        raise ValueError("Both dates are required for this test.")

    approval_gap = (sanction_date - recommended_date).days
    if approval_gap < 0:
        raise ValueError("Sanction date cannot precede recommended date.")

    print("\nLoading historical training data...")
    df = build_features()

    # Fit the same preprocessing and SVM pipeline on historical rows.
    results = {}

    for label_col, output_name in [
        ("time_overrun_label", "time_overrun"),
        ("underutilization_label", "underutilization"),
    ]:
        train = df.dropna(
            subset=FEATURES_NUM + FEATURES_CAT + [label_col]
        ).copy()

        X_num = train[FEATURES_NUM].to_numpy(dtype=float)
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
        X_cat = encoder.fit_transform(train[FEATURES_CAT])
        X = np.hstack([X_num, X_cat])

        y = train[label_col].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError(f"{label_col} contains fewer than two classes.")

        model = SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X, y)

        # Construct one new row using the historical feature definitions.
        raw = pd.DataFrame([{
            "agency_raw": agency,
            "work_raw": description,
            "state": state,
            "sanctioned_amount": amount,
            "sanction_date": sanction_date,
            "recommended_date": recommended_date,
        }])

        raw["agency_id"] = raw["agency_raw"].apply(_agency_id)
        raw["work_type"] = raw["work_raw"].apply(_work_type)

        # Obtain the prior-reliability feature using the project's own
        # helper. If it cannot calculate a value for this agency, fail
        # clearly rather than silently inventing a feature.
        raw = add_prior_reliability(raw, raw_col="agency_raw")

        raw["approval_gap_days"] = approval_gap
        raw["sanction_season"] = sanction_date.month
        raw["sanction_season"] = raw["sanction_season"].map(_SEASON_MAP)

        agency_rows = df[df["agency_id"] == raw.loc[0, "agency_id"]]

        # Match the all-time agency feature definitions.
        if agency_rows.empty:
            time_rate = df["time_overrun_label"].mean()
            under_rate = df["underutilization_label"].mean()
            print(
                f"\nNote: No exact historical agency match for "
                f"'{agency}'. Using dataset-wide rates as a cold-start fallback."
            )
        else:
            time_rate = agency_rows["time_overrun_label"].mean()
            under_rate = agency_rows["underutilization_label"].mean()

        raw["agency_alltime_time_rate"] = time_rate
        raw["agency_alltime_underutil_rate"] = under_rate

        missing = raw[FEATURES_NUM + FEATURES_CAT].isna().any()
        if missing.any():
            raise ValueError(
                "Missing required feature(s): "
                + ", ".join(missing[missing].index)
            )

        new_num = scaler.transform(raw[FEATURES_NUM].to_numpy(dtype=float))
        new_cat = encoder.transform(raw[FEATURES_CAT])
        new_X = np.hstack([new_num, new_cat])

        score = float(model.predict_proba(new_X)[0, 1] * 100)
        results[output_name] = (round(score, 2), risk_tier(score))

    print("\n===== MODULE 1 RESULTS =====")
    print(
        f"Time-overrun score: {results['time_overrun'][0]:.2f}/100 "
        f"({results['time_overrun'][1]})"
    )
    print(
        f"Under-utilization score: {results['underutilization'][0]:.2f}/100 "
        f"({results['underutilization'][1]})"
    )
    print("Model route: SVM")
    print("\nTest completed. No dashboard files or score CSVs were changed.")
    print(
        "Note: these are experimental single-work scores. "
        "The agency cold-start fallback and model refitting mean they "
        "are not guaranteed to match the batch CSV scores."
    )


if __name__ == "__main__":
    main()
