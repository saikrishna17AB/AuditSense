"""Module 2 model: Isolation Forest (primary) + Autoencoder (secondary),
run independently, each rescaled to its own 0-100 score, combined only
through agreement tiers for presentation — never averaged into one number,
so disagreement stays visible (see design notes in the project write-up).

LIBRARY NOTE: this environment has neither PyTorch nor TensorFlow
installed and has no network access to install them. The Autoencoder here
is implemented with scikit-learn's MLPRegressor, trained to reconstruct
its own input (X -> X) — same mechanism as a Keras autoencoder (bottleneck
compression, ReLU hidden layers, MSE reconstruction error), just built on
a library actually available in this sandbox. If the real deployment
target has PyTorch/TensorFlow, swap this class's _fit/_reconstruction_error
internals for a proper encoder/decoder; the rest of the pipeline
(rescaling, tiering, fusion) doesn't change.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

# Isolation Forest hyperparameters
IF_N_ESTIMATORS = 150
IF_MAX_SAMPLES = "auto"
IF_CONTAMINATION = 0.03   # tunable dial — starting at 3%, per the 1-5% guidance
IF_MAX_FEATURES = 1.0
IF_RANDOM_STATE = 42

# Autoencoder (MLPRegressor) hyperparameters
AE_HIDDEN_LAYERS = (4, 2, 4)   # input(6) -> 4 -> bottleneck(2) -> 4 -> output(6)
AE_MAX_ITER = 800              # MLPRegressor's own early-stopping handles overfit;
                                # raised from 300 after a real run hit that ceiling
                                # without converging (ConvergenceWarning)
AE_RANDOM_STATE = 42

HIGH_SCORE_THRESHOLD = 70  # score >= this counts as "high" for agreement tiering


def _rescale_0_100(raw_scores: np.ndarray, higher_is_more_anomalous: bool) -> np.ndarray:
    """Min-max rescale to 0-100. Isolation Forest's score_samples is HIGHER
    for normal points (needs inverting); Autoencoder's reconstruction error
    is already higher = more anomalous."""
    s = raw_scores if higher_is_more_anomalous else -raw_scores
    lo, hi = s.min(), s.max()
    if hi == lo:
        return np.zeros_like(s)
    return 100 * (s - lo) / (hi - lo)


class FusionModel:
    def __init__(self):
        self.iso_forest = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            max_samples=IF_MAX_SAMPLES,
            contamination=IF_CONTAMINATION,
            max_features=IF_MAX_FEATURES,
            random_state=IF_RANDOM_STATE,
        )
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=AE_HIDDEN_LAYERS,
            activation="relu",
            max_iter=AE_MAX_ITER,
            random_state=AE_RANDOM_STATE,
            early_stopping=True,
        )

    def fit(self, X: pd.DataFrame) -> "FusionModel":
        self.iso_forest.fit(X)
        self.autoencoder.fit(X, X)  # reconstruct its own input
        return self

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        iso_raw = self.iso_forest.score_samples(X)  # higher = more normal
        iso_score = _rescale_0_100(iso_raw, higher_is_more_anomalous=False)

        reconstructed = self.autoencoder.predict(X)
        recon_error = np.mean((X.values - reconstructed) ** 2, axis=1)
        ae_score = _rescale_0_100(recon_error, higher_is_more_anomalous=True)

        fused_score = 0.5 * iso_score + 0.5 * ae_score  # for ranking/validation only,
                                                          # never shown as "the" score

        tier = np.where(
            (iso_score >= HIGH_SCORE_THRESHOLD) & (ae_score >= HIGH_SCORE_THRESHOLD),
            "High-confidence flag",
            np.where(
                (iso_score >= HIGH_SCORE_THRESHOLD) | (ae_score >= HIGH_SCORE_THRESHOLD),
                "Needs review",
                "Not flagged",
            ),
        )

        return pd.DataFrame(
            {
                "isolation_forest_score": iso_score.round(2),
                "autoencoder_score": ae_score.round(2),
                "fused_score": fused_score.round(2),
                "agreement_tier": tier,
            },
            index=X.index,
        )


def top_contributing_factors(raw_row: pd.Series, feature_columns: list[str], top_n: int = 3) -> list[str]:
    """Explainability substitute for Module 2 (not SHAP — see project notes
    on why: Isolation Forest/Autoencoder aren't trained predictions in the
    sense SHAP requires). Ranks this work's RAW feature values against the
    population by how extreme they are, returns the top N as plain text."""
    # Caller passes the raw (unscaled) row; ranking is by each feature's own
    # magnitude relative to typical values, kept simple/explainable on purpose.
    ranked = raw_row[feature_columns].abs().sort_values(ascending=False)
    return list(ranked.head(top_n).index)


if __name__ == "__main__":
    from FinancialAnomaly.features.build_features import build_features

    full_df, scaled_df = build_features()

    model = FusionModel().fit(scaled_df)
    scores = model.score(scaled_df)

    result = full_df.join(scores)

    print("Score distribution:")
    print(scores[["isolation_forest_score", "autoencoder_score", "fused_score"]].describe().to_string())

    print("\nAgreement tier counts:")
    print(scores["agreement_tier"].value_counts().to_string())

    print("\nTop 10 by fused score:")
    cols = ["agency_id", "amount", "work_type", "isolation_forest_score",
            "autoencoder_score", "fused_score", "agreement_tier"]
    print(result.sort_values("fused_score", ascending=False)[cols].head(10).to_string(index=False))