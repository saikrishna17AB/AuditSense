"""Validation for Module 2. No fraud labels exist for MPLADS, so accuracy
in the normal supervised sense isn't measurable. Four checks instead:

1. Synthetic injection test — plant synthetic anomalies into a copy of the
   real data, measure whether the fused score ranks them highly (AUC,
   average precision, recall@N), broken down by anomaly type.
2. Ablation testing — remove one feature at a time, measure how much the
   top-N flagged set changes.
3. Score-distribution sanity check — real scores should show a small right
   tail, not a bimodal/uniform spread.
4. Cross-algorithm agreement rate — of everything either model flags in
   its own top N, what fraction do both agree on.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from FinancialAnomaly.features.build_features import build_raw_features, FEATURE_COLUMNS
from FinancialAnomaly.model.fusion_model import FusionModel

N_PER_TYPE = 60          # synthetic examples per anomaly type
N_TYPES = 5
TOP_N = 300              # matches the earlier recall@300 convention
RANDOM_STATE = 42

ANOMALY_TYPES = [
    "compound_threshold_fast",
    "compound_outlier_fast",
    "threshold_alone",
    "fast_alone",
    "outlier_alone",
]


def _make_synthetic(raw_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Builds synthetic anomalous feature rows by taking real rows as a
    base (so f2/f3/f4 stay realistic) and overriding f1/f5/f6 to represent
    each anomaly type. Returns a raw-feature dataframe, same columns as
    build_raw_features()[FEATURE_COLUMNS], plus 'anomaly_type'."""
    base = raw_df[FEATURE_COLUMNS].sample(
        n=N_PER_TYPE * N_TYPES, random_state=rng.integers(1e6)
    ).reset_index(drop=True)

    rows = []
    for i, anomaly_type in enumerate(ANOMALY_TYPES):
        chunk = base.iloc[i * N_PER_TYPE:(i + 1) * N_PER_TYPE].copy()
        if anomaly_type == "compound_threshold_fast":
            chunk["f1_threshold_avoidance"] = rng.uniform(0.9, 1.0, N_PER_TYPE)
            chunk["f5_approval_gap"] = rng.uniform(0.9, 1.0, N_PER_TYPE)
        elif anomaly_type == "compound_outlier_fast":
            chunk["f6_agency_fingerprint"] = rng.uniform(6, 10, N_PER_TYPE)
            chunk["f5_approval_gap"] = rng.uniform(0.9, 1.0, N_PER_TYPE)
        elif anomaly_type == "threshold_alone":
            chunk["f1_threshold_avoidance"] = rng.uniform(0.9, 1.0, N_PER_TYPE)
        elif anomaly_type == "fast_alone":
            chunk["f5_approval_gap"] = rng.uniform(0.9, 1.0, N_PER_TYPE)
        elif anomaly_type == "outlier_alone":
            chunk["f6_agency_fingerprint"] = rng.uniform(6, 10, N_PER_TYPE)
        chunk["anomaly_type"] = anomaly_type
        rows.append(chunk)

    return pd.concat(rows, ignore_index=True)


def injection_test(raw_df: pd.DataFrame, seed: int = RANDOM_STATE) -> dict:
    rng = np.random.default_rng(seed)
    synthetic = _make_synthetic(raw_df, rng)

    real_features = raw_df[FEATURE_COLUMNS].reset_index(drop=True)
    combined = pd.concat([real_features, synthetic[FEATURE_COLUMNS]], ignore_index=True)
    is_synthetic = np.array([0] * len(real_features) + [1] * len(synthetic))
    anomaly_type = np.array(["real"] * len(real_features) + list(synthetic["anomaly_type"]))

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(combined), columns=FEATURE_COLUMNS)

    model = FusionModel().fit(scaled)
    scores = model.score(scaled)

    results = {}
    for col, label in [
        ("isolation_forest_score", "Isolation Forest only"),
        ("autoencoder_score", "Autoencoder only"),
        ("fused_score", "Fused (both, α=0.5)"),
    ]:
        auc = roc_auc_score(is_synthetic, scores[col])
        ap = average_precision_score(is_synthetic, scores[col])
        top_idx = scores[col].nlargest(TOP_N).index
        recall_at_n = is_synthetic[top_idx].sum() / is_synthetic.sum()
        results[label] = {"AUC": round(auc, 3), "avg_precision": round(ap, 3),
                           f"recall@{TOP_N}": round(recall_at_n, 3)}

    top_idx = scores["fused_score"].nlargest(TOP_N).index
    recall_by_type = {}
    for t in ANOMALY_TYPES:
        type_mask = anomaly_type == t
        type_total = type_mask.sum()
        type_caught = (type_mask[top_idx]).sum() if type_total > 0 else 0
        recall_by_type[t] = round(type_caught / type_total, 3) if type_total > 0 else None

    return {
        "summary": results,
        "recall_by_type": recall_by_type,
        "scores": scores,
        "is_synthetic": is_synthetic,
    }


def ablation_test(raw_df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Baseline top-N flagged set vs. top-N with each feature dropped
    (replaced by its column mean, i.e. zero signal after standardizing),
    reported as % overlap with the baseline top-N set."""
    rng = np.random.default_rng(seed)
    synthetic = _make_synthetic(raw_df, rng)
    real_features = raw_df[FEATURE_COLUMNS].reset_index(drop=True)
    combined = pd.concat([real_features, synthetic[FEATURE_COLUMNS]], ignore_index=True)

    def _top_n_set(feature_df: pd.DataFrame) -> set:
        scaled = pd.DataFrame(
            StandardScaler().fit_transform(feature_df), columns=feature_df.columns
        )
        model = FusionModel().fit(scaled)
        scores = model.score(scaled)
        return set(scores["fused_score"].nlargest(TOP_N).index)

    baseline_set = _top_n_set(combined)

    rows = []
    for col in FEATURE_COLUMNS:
        dropped = combined.copy()
        dropped[col] = dropped[col].mean()  # neutralize this feature's signal
        dropped_set = _top_n_set(dropped)
        overlap = len(baseline_set & dropped_set) / TOP_N
        rows.append({"feature_removed": col, "pct_overlap_with_baseline": round(overlap, 3),
                     "pct_changed": round(1 - overlap, 3)})

    return pd.DataFrame(rows).sort_values("pct_changed", ascending=False)


def score_distribution_check(scores: pd.Series) -> dict:
    """A healthy anomaly-score distribution has most mass low with a small
    right tail — not bimodal, not uniform."""
    return {
        "median": round(scores.median(), 2),
        "p90": round(scores.quantile(0.90), 2),
        "p99": round(scores.quantile(0.99), 2),
        "max": round(scores.max(), 2),
        "pct_above_50": round((scores > 50).mean() * 100, 2),
        "pct_above_70": round((scores > 70).mean() * 100, 2),
    }


def cross_algorithm_agreement(scores: pd.DataFrame, top_n: int = TOP_N) -> float:
    iso_top = set(scores["isolation_forest_score"].nlargest(top_n).index)
    ae_top = set(scores["autoencoder_score"].nlargest(top_n).index)
    return round(len(iso_top & ae_top) / top_n, 3)


if __name__ == "__main__":
    raw_df = build_raw_features()

    print("=" * 70)
    print("1. SYNTHETIC INJECTION TEST")
    print("=" * 70)
    result = injection_test(raw_df)
    print(pd.DataFrame(result["summary"]).T.to_string())
    print(f"\nRecall by injected anomaly type (in top {TOP_N} by fused score):")
    for t, r in result["recall_by_type"].items():
        print(f"  {t}: {r}")

    print("\n" + "=" * 70)
    print("2. ABLATION TEST (feature removed -> % of top-300 that changed)")
    print("=" * 70)
    ablation = ablation_test(raw_df)
    print(ablation.to_string(index=False))

    print("\n" + "=" * 70)
    print("3. SCORE-DISTRIBUTION SANITY CHECK (real data only, fused score)")
    print("=" * 70)
    real_scores = result["scores"]["fused_score"][result["is_synthetic"] == 0]
    print(score_distribution_check(real_scores))

    print("\n" + "=" * 70)
    print("4. CROSS-ALGORITHM AGREEMENT RATE (top-300 overlap, real+synthetic combined)")
    print("=" * 70)
    print(f"Isolation Forest and Autoencoder agree on "
          f"{cross_algorithm_agreement(result['scores']) * 100:.1f}% of their respective top-{TOP_N}")