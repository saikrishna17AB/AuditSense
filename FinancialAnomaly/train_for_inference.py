
"""Train and save Module 2 artifacts for repeatable single-work inference.

Existing feature and model files remain unchanged.
Run from the project root:
    python -m FinancialAnomaly.train_for_inference
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from FinancialAnomaly.features.build_features import (
    FEATURE_COLUMNS,
    build_raw_features,
    add_rule_flags,
)
from FinancialAnomaly.model.fusion_model import FusionModel


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "Artifacts" / "module2" / "inference"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    print("Loading historical data and building six features...")

    # Compute features using the existing implementation.
    raw_df = build_raw_features()
    raw_df = add_rule_flags(raw_df)

    # Keep only rows with valid feature values.
    raw_df[FEATURE_COLUMNS] = raw_df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    valid_mask = raw_df[FEATURE_COLUMNS].notna().all(axis=1)
    raw_df = raw_df.loc[valid_mask].copy()

    if len(raw_df) < 100:
        raise RuntimeError(
            f"Not enough valid historical rows for training: {len(raw_df)}"
        )

    print(f"Historical rows used: {len(raw_df):,}")

    # Fit the scaler once on the reference population.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw_df[FEATURE_COLUMNS])
    X_scaled = pd.DataFrame(
        X_scaled,
        columns=FEATURE_COLUMNS,
        index=raw_df.index,
    )

    print("Training Isolation Forest and autoencoder...")
    model = FusionModel()
    model.fit(X_scaled)

    # Reference score distributions for later calibration.
    # Isolation Forest: lower score_samples means more anomalous.
    if_raw = model.iso_forest.score_samples(X_scaled)
    ae_reconstruction = model.autoencoder.predict(X_scaled)
    ae_errors = np.mean(
        (X_scaled.to_numpy() - ae_reconstruction) ** 2,
        axis=1,
    )

    # Save empirical reference distributions, rather than using the
    # min/max of each incoming test batch.
    reference = {
        "isolation_forest_anomaly": np.sort(-if_raw).tolist(),
        "autoencoder_error": np.sort(ae_errors).tolist(),
    }

    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(model, ARTIFACT_DIR / "fusion_model.joblib")

    with open(ARTIFACT_DIR / "reference_scores.json", "w", encoding="utf-8") as f:
        json.dump(reference, f)

    with open(ARTIFACT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "training_rows": int(len(raw_df)),
                "calibration_method": "empirical reference percentile",
                "score_interpretation": (
                    "Percentile indicates how extreme a score is relative "
                    "to the training reference distribution; it is not a "
                    "probability of fraud."
                ),
            },
            f,
            indent=2,
        )

    print("\nSaved inference artifacts to:")
    print(ARTIFACT_DIR)
    print("  scaler.joblib")
    print("  fusion_model.joblib")
    print("  reference_scores.json")
    print("  metadata.json")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
