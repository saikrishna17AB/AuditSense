"""Trains Module 2's fusion model ONCE and saves it, so the dashboard's
live 'Score a New Work' endpoint (Dashboard/app.py's assess_module2()) can
load a ready-made model instead of retraining on every request.
"""
from pathlib import Path
import sys
import json
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from FinancialAnomaly.features.build_features import build_raw_features, add_rule_flags, FEATURE_COLUMNS
from FinancialAnomaly.model.fusion_model import FusionModel


def main():
    output_dir = PROJECT_ROOT / "Artifacts" / "module2" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building Module 2 features...")
    full_df = build_raw_features()
    full_df = add_rule_flags(full_df)

    # Fit and KEEP the scaler — build_features() normally discards it.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(full_df[FEATURE_COLUMNS])
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=full_df.index)

    print(f"Training FusionModel on {len(scaled_df):,} works...")
    model = FusionModel().fit(scaled_df)

    # Raw (pre-rescaling) anomaly values for every historical work.
    if_raw = model.iso_forest.score_samples(scaled_df)
    if_anomaly_values = -if_raw

    reconstructed = model.autoencoder.predict(scaled_df)
    ae_error_values = np.mean((scaled_df.to_numpy() - reconstructed) ** 2, axis=1)

    # np.searchsorted (used by assess_module2()'s percentile lookup)
    # requires a SORTED ascending array.
    reference_scores = {
        "isolation_forest_anomaly": sorted(float(v) for v in if_anomaly_values),
        "autoencoder_error": sorted(float(v) for v in ae_error_values),
    }

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_works": int(len(full_df)),
        "feature_columns": FEATURE_COLUMNS,
    }

    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(model, output_dir / "fusion_model.joblib")
    with open(output_dir / "reference_scores.json", "w", encoding="utf-8") as f:
        json.dump(reference_scores, f)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {output_dir} — restart the Flask dashboard server if it's already running.")


if __name__ == "__main__":
    main()