"""Predict-and-save pipeline for Module 1 (Time Overrun + Under-utilization).

time_overrun_model.py's train_and_evaluate() only reports train/test AUC —
it never saves a per-work score. This module adds that: trains on ALL
available labeled data (once the held-out AUC above has already told you
the model generalizes), scores every work, and saves one row per work_key
— the shape the dashboard needs to join against Module 2/3's output.

TENSORFLOW NOTE: this sandbox has no TensorFlow (confirmed: `import
tensorflow` fails here). The ANN segments ('large' work_type groups,
>= MIN_SEGMENT_SIZE works) are architecturally real and will run on a
machine that has TF installed (it's in requirements.txt) — guarded
imports here mean this script still produces real SVM-based scores for
EVERY work in this environment, falling back ALL segments to SVM rather
than silently dropping the ANN segment's works. When run somewhere with
TF available, routing switches back to the real ANN+SVM split and the
routing documented in time_overrun_model.py is used as originally
designed.
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from Time_Cost_Overrun.leakage_safe.build_features import build_features, FEATURES_NUM, FEATURES_CAT

MIN_SEGMENT_SIZE = 150

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

LABELS = {
    "time_overrun": "time_overrun_label",
    "underutilization": "underutilization_label",
}

LOW, MED = 40, 70  # risk tier cutoffs, matching Module 2's Low/Medium/High convention


def _risk_tier(score_0_100: float) -> str:
    if score_0_100 < LOW:
        return "Low"
    if score_0_100 < MED:
        return "Medium"
    return "High"


def _encode(df: pd.DataFrame, cat_encoder=None, scaler=None, fit=False):
    X_num = df[FEATURES_NUM].to_numpy(dtype=float)
    if fit:
        scaler = StandardScaler().fit(X_num)
    X_num = scaler.transform(X_num)
    if fit:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[FEATURES_CAT])
    X_cat = cat_encoder.transform(df[FEATURES_CAT])
    return np.hstack([X_num, X_cat]), cat_encoder, scaler


def _build_ann(input_dim):
    inp = keras.Input(shape=(input_dim,))
    h = keras.layers.Dense(16, activation="relu")(inp)
    h = keras.layers.Dense(8, activation="relu")(h)
    out = keras.layers.Dense(1, activation="sigmoid")(h)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def _fit_predict_one_label(df: pd.DataFrame, label_col: str, seed: int = 42) -> pd.Series:
    """Fits on ALL rows (not train/test — this is the final scoring model,
    not the validation run) and returns a 0-100 probability-based score
    per row, indexed the same as df. Routes by segment_route, same logic
    as time_overrun_model.py, but falls back entirely to SVM if TF isn't
    importable in this environment."""
    counts = df["work_type"].value_counts()
    large_types = counts[counts >= MIN_SEGMENT_SIZE].index
    route = pd.Series(
        np.where(df["work_type"].isin(large_types), "ANN", "SVM"),
        index=df.index,
    )
    if not TF_AVAILABLE:
        route[:] = "SVM"  # entire dataset falls back — see module docstring

    scores = pd.Series(index=df.index, dtype=float)

    ann_mask = route == "ANN"
    if ann_mask.any():
        sub = df[ann_mask]
        X, enc, scaler = _encode(sub, fit=True)
        y = sub[label_col].to_numpy()
        tf.keras.utils.set_random_seed(seed)
        ann = _build_ann(X.shape[1])
        ann.fit(X, y, epochs=30, batch_size=64, verbose=0,
                class_weight={0: 1.0, 1: (y == 0).sum() / max((y == 1).sum(), 1)})
        scores.loc[sub.index] = ann.predict(X, verbose=0).ravel()

    svm_mask = route == "SVM"
    if svm_mask.any():
        sub = df[svm_mask]
        X, enc, scaler = _encode(sub, fit=True)
        y = sub[label_col].to_numpy()
        svm = SVC(kernel="rbf", probability=True, random_state=seed, class_weight="balanced")
        svm.fit(X, y)
        scores.loc[sub.index] = svm.predict_proba(X)[:, 1]

    return (scores * 100).round(2), route


def predict_all(seed: int = 42) -> pd.DataFrame:
    df = build_features()

    result = df[["work_key", "work_raw", "agency_raw", "agency_id", "state",
                 "work_type", "sanctioned_amount", "sanction_date"]].copy()

    for name, label_col in LABELS.items():
        score, route = _fit_predict_one_label(df, label_col, seed=seed)
        result[f"{name}_score"] = score
        result[f"{name}_tier"] = score.apply(_risk_tier)
        result[f"{name}_route"] = route

    if not TF_AVAILABLE:
        print("[predict_all] WARNING: TensorFlow not available in this environment — "
              "ALL works routed to SVM (ANN segments not run). Re-run on a machine with "
              "TensorFlow installed for the real ANN+SVM split.")

    return result


if __name__ == "__main__":
    out = predict_all()
    print(f"Scored {len(out)} works\n")

    for name in LABELS:
        print(f"=== {name} ===")
        print(f"Route counts: {out[f'{name}_route'].value_counts().to_dict()}")
        print(f"Tier counts: {out[f'{name}_tier'].value_counts().to_dict()}")
        print(f"Score distribution:\n{out[f'{name}_score'].describe().to_string()}\n")

    print("Sample rows:")
    cols = ["work_key", "agency_id", "work_type", "sanctioned_amount",
            "time_overrun_score", "time_overrun_tier",
            "underutilization_score", "underutilization_tier"]
    print(out[cols].head(10).to_string(index=False))

    out.to_csv("module1_scores.csv", index=False)
    print("\nSaved to module1_scores.csv")