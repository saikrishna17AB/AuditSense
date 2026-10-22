from functools import lru_cache
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from rapidfuzz.fuzz import ratio

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
MODULE3_DIR = PROJECT_ROOT / "Artifacts" / "module3"
CLEAN_DATA = MODULE3_DIR / "mplads_clean.csv"
EMBEDDING_FILE = MODULE3_DIR / "sbert_embeddings.npy"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)

# Match the weights used by Module 3's existing scoring pipeline.
TEXT_WEIGHT = 0.50
GEO_WEIGHT = 0.20
TIME_WEIGHT = 0.15
STRUCT_WEIGHT = 0.15

POTENTIAL_DUPLICATE = 0.75
NEEDS_REVIEW = 0.55

GEO_WEIGHTS = {
    "STATE": 0.10,
    "CONSTITUENCY": 0.15,
    "CITY": 0.10,
    "WARD": 0.10,
    "BLOCK": 0.20,
    "VILLAGE": 0.35,
}

STRUCTURED_FIELDS = ["CATEGORY", "IDA APPROVAL", "HOUSE"]

# Load the reference data once per server process.
@lru_cache(maxsize=1)
def load_reference_data():
    if not CLEAN_DATA.exists():
        raise FileNotFoundError(f"Reference dataset not found: {CLEAN_DATA}")
    if not EMBEDDING_FILE.exists():
        raise FileNotFoundError(f"Embeddings not found: {EMBEDDING_FILE}")

    df = pd.read_csv(CLEAN_DATA)
    embeddings = np.load(EMBEDDING_FILE, mmap_mode="r")

    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"Dataset/embedding mismatch: {len(df)} rows, "
            f"{embeddings.shape[0]} embeddings"
        )
    if embeddings.ndim != 2 or embeddings.shape[1] != 384:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    return df, embeddings


@lru_cache(maxsize=1)
def load_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def clean_value(value):
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() in {"nan", "none", "null", "nat"}:
        return ""
    return value


def normalized_value(value):
    return clean_value(value).lower()


def build_work_text(work):
    # Keep the same field order and labels as the existing embedding pipeline.
    fields = [
        ("WORK", work.get("WORK")),
        ("CATEGORY", work.get("CATEGORY")),
        ("STATE", work.get("STATE")),
        ("CONSTITUENCY", work.get("CONSTITUENCY")),
        ("CITY", work.get("CITY")),
        ("BLOCK", work.get("BLOCK")),
        ("VILLAGE", work.get("VILLAGE")),
    ]

    parts = []
    for field, value in fields:
        value = clean_value(value)
        if value:
            if field == "WORK":
                parts.append(value)
            else:
                parts.append(f"{field.title()}: {value}")

    return " | ".join(parts)


def calculate_geo_similarity(new_work, existing_work):
    similarities = []
    weights = []

    for field, weight in GEO_WEIGHTS.items():
        a = normalized_value(new_work.get(field))
        b = normalized_value(existing_work.get(field))

        if not a or not b:
            continue

        if field == "VILLAGE":
            similarity = ratio(a, b) / 100.0
        else:
            similarity = 1.0 if a == b else 0.0

        similarities.append(similarity)
        weights.append(weight)

    if not weights:
        return 0.0

    return float(np.average(similarities, weights=weights))


def calculate_time_similarity(date_a, date_b):
    date_a = pd.to_datetime(date_a, errors="coerce")
    date_b = pd.to_datetime(date_b, errors="coerce")

    if pd.isna(date_a) or pd.isna(date_b):
        return 0.0

    delta_days = abs((date_a - date_b).days)
    return float(np.exp(-delta_days / 180.0))


def calculate_structured_similarity(new_work, existing_work):
    similarities = []

    for field in STRUCTURED_FIELDS:
        a = normalized_value(new_work.get(field))
        b = normalized_value(existing_work.get(field))

        if a and b:
            similarities.append(1.0 if a == b else 0.0)

    if not similarities:
        return 0.0

    return float(np.mean(similarities))


def score_tier(score):
    if score >= POTENTIAL_DUPLICATE:
        return "Potential Duplicate"
    if score >= NEEDS_REVIEW:
        return "Needs Review"
    return "Lower Likelihood"


def assess_financial_rules(payload):
    """
    Transparent first-pass checks for a newly submitted work.
    Historical-data-dependent checks remain unavailable unless
    their required inputs and baselines are integrated.
    """
    raw_amount = clean_value(payload.get("amount"))
    amount = None

    if raw_amount:
        try:
            amount = float(raw_amount)
            if not np.isfinite(amount) or amount < 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("Amount must be a valid non-negative number.")

    # The existing Module 2 rule checks works close to these thresholds.
    thresholds = [1500000, 2500000]
    threshold_avoidance = None

    if amount is not None:
        threshold_avoidance = any(
            0 < threshold - amount <= 50000
            for threshold in thresholds
        )

    return {
        "status": "partial",
        "checks": [
            {
                "name": "Threshold avoidance",
                "status": (
                    "flagged" if threshold_avoidance
                    else "not_flagged"
                ) if threshold_avoidance is not None else "unavailable",
                "value": threshold_avoidance,
                "explanation": (
                    "The entered amount is within ₹50,000 below "
                    "₹15 lakh or ₹25 lakh."
                    if threshold_avoidance
                    else (
                        "The amount did not meet the configured "
                        "threshold-avoidance rule."
                        if threshold_avoidance is False
                        else "Enter an amount to run this check."
                    )
                )
            },
            {
                "name": "Fast approval",
                "status": "unavailable",
                "value": None,
                "explanation": (
                    "A sanction date is required to calculate the "
                    "approval gap. The current form provides only "
                    "the recommended date."
                )
            },
            {
                "name": "Pricing outlier",
                "status": "unavailable",
                "value": None,
                "explanation": (
                    "This check requires a validated historical "
                    "amount baseline for the implementing agency. "
                    "It has not been calculated for this new work."
                )
            },
            {
                "name": "Benford analysis",
                "status": "unavailable",
                "value": None,
                "explanation": (
                    "Benford analysis requires a suitable group of "
                    "historical transactions; a single proposed "
                    "work is insufficient."
                )
            }
        ],
        "notice": (
            "These are preliminary rule checks, not a financial "
            "anomaly model prediction or a finding of misconduct."
        )
    }



def assess_duplicates(payload):
    df, embeddings = load_reference_data()

    description = clean_value(payload.get("description"))
    state = clean_value(payload.get("state"))

    if not description:
        raise ValueError("Work description is required.")
    if not state:
        raise ValueError("State is required.")

    new_work = {
        "WORK": description,
        "CATEGORY": clean_value(payload.get("category")),
        "STATE": state,
        "CONSTITUENCY": clean_value(payload.get("constituency")),
        "CITY": clean_value(payload.get("city")),
        "WARD": clean_value(payload.get("ward")),
        "BLOCK": clean_value(payload.get("block")),
        "VILLAGE": clean_value(payload.get("village")),
        "RECOMMENDED DATE": clean_value(payload.get("recommended_date")),
        "IDA APPROVAL": clean_value(payload.get("ida_approval")),
        "HOUSE": clean_value(payload.get("house")),
    }

    # Generate an embedding for this new work.
    text = build_work_text(new_work)
    model = load_sentence_model()
    new_embedding = np.asarray(
        model.encode([text], normalize_embeddings=True),
        dtype=np.float32,
    )[0]

    # Embeddings are normalized, so dot product equals cosine similarity.
    # Calculate similarities in chunks to limit temporary memory use.
    text_scores = np.empty(len(df), dtype=np.float32)
    batch_size = 5000

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = np.asarray(embeddings[start:end], dtype=np.float32)
        text_scores[start:end] = batch @ new_embedding

    # Shortlist the strongest text matches across the whole dataset.
    shortlist_size = min(500, len(df))
    if shortlist_size == 0:
        return {
            "matches": [],
            "reference_count": 0,
            "message": "The reference dataset contains no records.",
        }

    top_ids = np.argpartition(
        text_scores, -shortlist_size
    )[-shortlist_size:]

    matches = []
    for idx in top_ids:
        existing = df.iloc[int(idx)].to_dict()

        s_text = float(text_scores[idx])
        s_geo = calculate_geo_similarity(new_work, existing)
        s_time = calculate_time_similarity(
            new_work.get("RECOMMENDED DATE"),
            existing.get("RECOMMENDED DATE"),
        )
        s_struct = calculate_structured_similarity(new_work, existing)

        final_score = (
            TEXT_WEIGHT * s_text
            + GEO_WEIGHT * s_geo
            + TIME_WEIGHT * s_time
            + STRUCT_WEIGHT * s_struct
        )

        matches.append({
            "work_id": int(idx),
            "work": clean_value(existing.get("WORK")),
            "state": clean_value(existing.get("STATE")),
            "constituency": clean_value(existing.get("CONSTITUENCY")),
            "category": clean_value(existing.get("CATEGORY")),
            "city": clean_value(existing.get("CITY")),
            "block": clean_value(existing.get("BLOCK")),
            "village": clean_value(existing.get("VILLAGE")),
            "recommended_date": clean_value(
                existing.get("RECOMMENDED DATE")
            ),
            "text_similarity": round(s_text, 4),
            "geo_similarity": round(s_geo, 4),
            "time_similarity": round(s_time, 4),
            "structured_similarity": round(s_struct, 4),
            "duplicate_score": round(float(final_score), 4),
            "tier": score_tier(float(final_score)),
        })

    matches.sort(key=lambda item: item["duplicate_score"], reverse=True)

    return {
        "reference_count": len(df),
        "candidates_scored": len(matches),
        "matches": matches[:10],
        "scoring": {
            "formula": "0.50*S_text + 0.20*S_geo + 0.15*S_time + 0.15*S_struct",
            "potential_duplicate_threshold": POTENTIAL_DUPLICATE,
            "needs_review_threshold": NEEDS_REVIEW,
        },
        "limitations": [
            "These are ranked candidates, not confirmed duplicate works.",
            "Only the 500 strongest text-similarity matches were fully scored.",
            "Missing location, date, or structured fields can reduce assessment quality.",
            "Human verification is required before treating any match as a duplicate.",
        ],
    }


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "MPLADS Guardian Dashboard",
    })


@app.get("/api/kpis")
def kpis():
    path = DATA_DIR / "kpi_summary.json"
    if not path.exists():
        return jsonify({"error": "Dashboard data not built yet"}), 503
    return send_from_directory(DATA_DIR, path.name)


@app.get("/api/module1")
def module1():
    path = DATA_DIR / "module1_data.json"
    if not path.exists():
        return jsonify({"error": "Module 1 data not built yet"}), 503
    return send_from_directory(DATA_DIR, path.name)


@app.get("/api/module2")
def module2():
    path = DATA_DIR / "module2_data.json"
    if not path.exists():
        return jsonify({"error": "Module 2 data not built yet"}), 503
    return send_from_directory(DATA_DIR, path.name)


@app.get("/api/module3")
def module3():
    path = DATA_DIR / "module3_pairs.json"
    if not path.exists():
        return jsonify({"error": "Module 3 data not built yet"}), 503
    return send_from_directory(DATA_DIR, path.name)


# Module 1: quick historical-reference screening for a new work.
# This is a reference-based estimate, NOT a fresh ML model prediction.

MODULE1_SCORES = PROJECT_ROOT / "Artifacts" / "module1" / "module1_scores.csv"


# def assess_module1_reference(payload):
#     from rapidfuzz.fuzz import ratio

#     if not MODULE1_SCORES.exists():
#         raise FileNotFoundError(
#             f"Module 1 scores not found: {MODULE1_SCORES}"
#         )

#     history = pd.read_csv(MODULE1_SCORES, low_memory=False)

#     description = clean_value(
#         payload.get("WORK")
#         or payload.get("work")
#         or payload.get("description")
#         or payload.get("work_description")
#     )
#     state = normalized_value(
#         payload.get("STATE") or payload.get("state")
#     )
#     work_type = normalized_value(
#         payload.get("CATEGORY")
#         or payload.get("category")
#         or payload.get("work_type")
#         or description
#     )

#     if not description and not work_type:
#         return {
#             "status": "unavailable",
#             "message": "Enter a work description to compare historical works."
#         }

#     history["_state_match"] = (
#         history["state"].fillna("").astype(str).str.lower().str.strip()
#         == state
#     ) if state else False

#     history["_text_similarity"] = history["work_type"].fillna("").astype(
#         str
#     ).apply(lambda value: ratio(work_type, value.lower()) / 100.0)

#     # Prefer same-state works, but fall back to text similarity if needed.
#     same_state = history[history["_state_match"]]
#     pool = same_state if not same_state.empty else history

#     pool = pool.copy()
#     pool["_text_similarity"] = pool["work_type"].fillna("").astype(
#         str
#     ).apply(lambda value: ratio(work_type, value.lower()) / 100.0)

#     amount = payload.get("amount") or payload.get("sanctioned_amount")
#     try:
#         amount = float(amount) if amount not in (None, "") else None
#     except (TypeError, ValueError):
#         amount = None

#     if amount is not None and amount > 0:
#         historical_amount = pd.to_numeric(
#             pool["sanctioned_amount"], errors="coerce"
#         ).clip(lower=1)
#         pool["_amount_similarity"] = (
#             1 - (historical_amount - amount).abs()
#             / historical_amount.clip(lower=amount).clip(lower=1)
#         ).clip(lower=0)
#         pool["_amount_similarity"] = pool["_amount_similarity"].fillna(0)
#         pool["_match_score"] = (
#             0.75 * pool["_text_similarity"]
#             + 0.25 * pool["_amount_similarity"]
#         )
#     else:
#         pool["_match_score"] = pool["_text_similarity"]

#     best = pool.sort_values("_match_score", ascending=False).head(5)
#     best = best[best["_match_score"] > 0]

#     if best.empty:
#         return {
#             "status": "unavailable",
#             "message": "No sufficiently similar historical works were found."
#         }

#     def weighted_average(column):
#         values = pd.to_numeric(best[column], errors="coerce")
#         weights = best["_match_score"]
#         valid = values.notna() & weights.notna() & (weights > 0)
#         if not valid.any():
#             return None
#         return round(float((values[valid] * weights[valid]).sum()
#                            / weights[valid].sum()), 2)

#     return {
#         "status": "reference_estimate",
#         "method": "similar_historical_works",
#         "time_overrun_score": weighted_average("time_overrun_score"),
#         "underutilization_score": weighted_average("underutilization_score"),
#         "similar_works_used": int(len(best)),
#         "best_match_similarity": round(float(best["_match_score"].max()), 3),
#         "evidence": [
#             {
#                 "work": clean_value(row.get("work_raw")),
#                 "state": clean_value(row.get("state")),
#                 "similarity": round(float(row["_match_score"]), 3),
#                 "time_overrun_score": clean_value(row.get("time_overrun_score")),
#                 "underutilization_score": clean_value(row.get("underutilization_score"))
#             }
#             for _, row in best.iterrows()
#         ],
#         "notice": (
#             "These are estimates derived from similar historical works, "
#             "not predictions generated by the Module 1 ML models."
#         )
#     }


def assess_module1_ml(payload):
    """Run Module 1 ML models against one newly entered work.

    This follows the feature preparation and SVM approach used by
    Time_Cost_Overrun/test_single_work.py.
    """
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
    from Time_Cost_Overrun.features.prior_history import (
        add_prior_reliability,
    )

    amount_raw = payload.get("amount") or payload.get("sanctioned_amount")
    try:
        amount = float(amount_raw)
    except (TypeError, ValueError):
        raise ValueError("Enter a valid positive amount for Module 1.")

    if not np.isfinite(amount) or amount <= 0:
        raise ValueError("The Module 1 amount must be greater than zero.")

    state = clean_value(payload.get("state") or payload.get("STATE"))
    agency = clean_value(
        payload.get("agency")
        or payload.get("agency_name")
        or payload.get("agency_raw")
    )
    description = clean_value(
        payload.get("description")
        or payload.get("work")
        or payload.get("work_description")
        or payload.get("WORK")
    )

    if not state or not agency or not description:
        raise ValueError(
            "Module 1 requires state, agency, and work description."
        )

    def parse_date(value, field_name):
        value = clean_value(value)
        if not value:
            raise ValueError(
                f"Module 1 requires the {field_name}."
            )

        # parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
        parsed = pd.to_datetime(value, errors="coerce", format="ISO8601")
        if pd.isna(parsed):
            raise ValueError(f"Invalid {field_name}: {value}")
        return parsed

    recommended_date = parse_date(
        payload.get("recommended_date"), "recommended date"
    )
    sanction_date = parse_date(
        payload.get("sanction_date"), "sanction date"
    )

    if sanction_date < recommended_date:
        raise ValueError(
            "Sanction date cannot precede recommended date."
        )

    # Load historical training data using the module's own feature builder.
    history = build_features()

    # Prepare the new work using the same feature definitions as the
    # standalone single-work test.
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
    raw = add_prior_reliability(raw, raw_col="agency_raw")

    raw["approval_gap_days"] = (
        sanction_date - recommended_date
    ).days
    raw["sanction_season"] = _SEASON_MAP.get(sanction_date.month)

    # Match the historical agency-rate features. For a new agency, use
    # dataset-wide historical label rates as the cold-start fallback.
    agency_rows = history[
        history["agency_id"] == raw.loc[0, "agency_id"]
    ]

    if agency_rows.empty:
        time_rate = float(history["time_overrun_label"].mean())
        under_rate = float(history["underutilization_label"].mean())
        cold_start = True
    else:
        time_rate = float(agency_rows["time_overrun_label"].mean())
        under_rate = float(agency_rows["underutilization_label"].mean())
        cold_start = False

    raw["agency_alltime_time_rate"] = time_rate
    raw["agency_alltime_underutil_rate"] = under_rate

    results = {}

    targets = [
        ("time_overrun_label", "time_overrun"),
        ("underutilization_label", "underutilization"),
    ]

    for label_column, output_name in targets:
        train = history.dropna(
            subset=FEATURES_NUM + FEATURES_CAT + [label_column]
        ).copy()

        if train.empty or train[label_column].nunique() < 2:
            raise ValueError(
                f"Insufficient training data for {output_name} model."
            )

        # Fit preprocessing and model using historical rows.
        scaler = StandardScaler()
        X_num = scaler.fit_transform(
            train[FEATURES_NUM].to_numpy(dtype=float)
        )

        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        except TypeError:
            # Compatibility with older scikit-learn versions.
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
            )

        X_cat = encoder.fit_transform(train[FEATURES_CAT])
        X_train = np.hstack([X_num, X_cat])
        y_train = train[label_column].astype(int).to_numpy()

        model = SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

        missing = raw[FEATURES_NUM + FEATURES_CAT].isna().any()
        if missing.any():
            names = ", ".join(missing[missing].index)
            raise ValueError(
                f"Module 1 cannot calculate required features: {names}"
            )

        new_num = scaler.transform(
            raw[FEATURES_NUM].to_numpy(dtype=float)
        )
        new_cat = encoder.transform(raw[FEATURES_CAT])
        X_new = np.hstack([new_num, new_cat])

        # This is the model's estimated positive-class score on a 0–100
        # scale; it is not guaranteed to be a calibrated probability.
        score = float(model.predict_proba(X_new)[0, 1] * 100)

        tier = (
            "Low" if score < 40
            else "Medium" if score < 70
            else "High"
        )

        results[output_name] = {
            "score": round(score, 2),
            "tier": tier,
        }

    return {
        "status": "completed",
        "method": "svm_single_work_inference",
        "time_overrun_score": results["time_overrun"]["score"],
        "time_overrun_tier": results["time_overrun"]["tier"],
        "underutilization_score": results["underutilization"]["score"],
        "underutilization_tier": results["underutilization"]["tier"],
        "model": "SVM",
        "cold_start_agency": cold_start,
        "notice": (
            "Scores are generated by SVM models fitted on historical "
            "training data during this assessment. They are model scores, "
            "not guaranteed calibrated probabilities. For production, "
            "save and load versioned fitted models instead of refitting "
            "on every request."
        ),
    }


# Module 2: saved financial-anomaly model inference.
MODULE2_ARTIFACT_DIR = (
    PROJECT_ROOT / "Artifacts" / "module2" / "inference"
)


def assess_module2(payload):
    import json
    import joblib

    from FinancialAnomaly.features.build_features import (
        FEATURE_COLUMNS,
        build_raw_features,
        add_rule_flags,
    )
    from FinancialAnomaly.shared.data_loader import load_sanctioned_works

    required = [
        "scaler.joblib",
        "fusion_model.joblib",
        "reference_scores.json",
        "metadata.json",
    ]
    missing = [
        name for name in required
        if not (MODULE2_ARTIFACT_DIR / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Module 2 inference artifacts missing: {missing}"
        )

    amount_raw = payload.get("amount") or payload.get("sanctioned_amount")
    try:
        amount = float(amount_raw)
    except (TypeError, ValueError):
        raise ValueError("Enter a valid positive work amount.")

    if not np.isfinite(amount) or amount <= 0:
        raise ValueError("Work amount must be greater than zero.")

    state = clean_value(payload.get("state") or payload.get("STATE"))
    agency = clean_value(
        payload.get("agency") or payload.get("agency_name")
        or payload.get("agency_raw")
    )
    description = clean_value(
        payload.get("description") or payload.get("work")
        or payload.get("work_description")
    )

    if not state or not agency or not description:
        raise ValueError(
            "State, agency name, and work description are required "
            "for Module 2 assessment."
        )

    def parse_date(value, field_name):
        value = clean_value(value)
        if not value:
            return pd.NaT

        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
        if pd.isna(parsed):
            raise ValueError(
                f"Invalid {field_name}. Use a valid date such as "
                "2026-10-08 or 08-Oct-2026."
            )
        return parsed

    rec_date = parse_date(
        payload.get("recommended_date"), "recommended date"
    )
    san_date = parse_date(
        payload.get("sanction_date"), "sanction date"
    )

    if (
        pd.notna(rec_date)
        and pd.notna(san_date)
        and san_date < rec_date
    ):
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

    # Reuse the same feature pipeline as the working single-work test.
    history = load_sanctioned_works()
    combined = pd.concat(
        [history.copy(), pd.DataFrame([new_row])],
        ignore_index=True,
        sort=False,
    )
    scored = build_raw_features(combined)
    scored = add_rule_flags(scored)

    row = scored.iloc[[-1]].copy()
    feature_row = row.iloc[0]
    raw_values = row[list(FEATURE_COLUMNS)].replace(
        [np.inf, -np.inf], np.nan
    )

    if raw_values.isna().any(axis=None):
        missing_features = raw_values.columns[
            raw_values.isna().any()
        ].tolist()
        return {
            "status": "unavailable",
            "message": (
                "Could not calculate all Module 2 features: "
                + ", ".join(missing_features)
            ),
        }

    scaler = joblib.load(MODULE2_ARTIFACT_DIR / "scaler.joblib")
    model = joblib.load(MODULE2_ARTIFACT_DIR / "fusion_model.joblib")

    with open(
        MODULE2_ARTIFACT_DIR / "reference_scores.json",
        encoding="utf-8",
    ) as f:
        reference = json.load(f)

    # Keep feature names for compatibility with the saved scaler/model.
    X_scaled = scaler.transform(raw_values[list(FEATURE_COLUMNS)])
    X_scaled_df = pd.DataFrame(
        X_scaled, columns=list(FEATURE_COLUMNS)
    )

    if_raw = float(model.iso_forest.score_samples(X_scaled_df)[0])
    if_anomaly = -if_raw

    reconstructed = model.autoencoder.predict(X_scaled_df)
    ae_error = float(np.mean(
        (X_scaled_df.to_numpy() - reconstructed) ** 2, axis=1
    )[0])

    def percentile(value, reference_values):
        ref = np.asarray(reference_values, dtype=float)
        if len(ref) == 0:
            return None
        return round(
            float(100 * np.searchsorted(
                ref, value, side="right"
            ) / len(ref)),
            2,
        )

    if_pct = percentile(
        if_anomaly, reference.get("isolation_forest_anomaly", [])
    )
    ae_pct = percentile(
        ae_error, reference.get("autoencoder_error", [])
    )

    if if_pct is None or ae_pct is None:
        raise ValueError(
            "Module 2 reference-score calibration data is empty."
        )

    fused_pct = round((if_pct + ae_pct) / 2, 2)

    # Evidence-based explanations from the module's raw feature values.
    evidence = []
    evidence_rules = [
        (
            "f1_threshold_avoidance",
            lambda x: x > 0,
            "Amount is close to a configured scrutiny threshold."
        ),
        (
            "f2_benford_excess",
            lambda x: x > 0,
            "Agency Benford deviation exceeds the dataset baseline."
        ),
        (
            "f3_round_number",
            lambda x: x >= 0.6,
            "Amount has a strong round-number pattern."
        ),
        (
            "f4_agency_cost_trend",
            lambda x: x > 0,
            "Agency monthly average cost trend is positive."
        ),
        (
            "f5_approval_gap",
            lambda x: x > 0,
            "Recommendation-to-sanction gap is unusually short."
        ),
        (
            "f6_agency_fingerprint",
            lambda x: x > 3,
            "Amount is more than 3 agency standard deviations "
            "from its historical mean."
        ),
    ]

    feature_values = {}
    for name in FEATURE_COLUMNS:
        value = float(feature_row[name])
        feature_values[name] = round(value, 4)

    for column, condition, message in evidence_rules:
        value = float(feature_row[column])
        if condition(value):
            evidence.append({
                "feature": column,
                "value": round(value, 4),
                "explanation": message,
            })

    rule_flags = {}
    for column in (
        "flag_threshold_avoidance",
        "flag_fast_approval",
        "flag_pricing_outlier",
    ):
        value = feature_row.get(column, False)
        rule_flags[column] = (
            bool(value) if pd.notna(value) else False
        )

    return {
        "status": "completed",
        "method": "saved_fusion_model_and_reference_percentiles",
        "isolation_forest_anomaly_value": round(if_anomaly, 6),
        "isolation_forest_percentile": if_pct,
        "autoencoder_reconstruction_error": round(ae_error, 6),
        "autoencoder_percentile": ae_pct,
        "fused_reference_percentile": fused_pct,
        "features": feature_values,
        "rule_flags": rule_flags,
        "evidence": evidence,
        "notice": (
            "Reference percentiles measure unusualness relative to "
            "historical model scores. They are not probabilities of fraud. "
            "This prototype recomputes features on historical data plus "
            "the new work, so some baselines may be influenced by the new row."
        ),
    }



@app.post("/api/assess")
def assess_new_work():
    payload = request.get_json(silent=True)

    if payload is None:
        payload = request.form.to_dict()

    try:
        # Module 3: duplicate-work screening.
        result = assess_duplicates(payload)

        # Module 2: saved financial-anomaly model.
        try:
            module2_result = assess_module2(payload)
        except FileNotFoundError:
            module2_result = {
                "status": "unavailable",
                "message": "Module 2 inference artifacts or historical data are missing."
            }

        result["module2_screening"] = module2_result

        # Keep the transparent preliminary rule checks as a separate result.
        result["financial_screening"] = assess_financial_rules(payload)

        # Module 1: historical-reference estimate.
        try:
            # module1_result = assess_module1_reference(payload)
            module1_result = assess_module1_ml(payload)
        except FileNotFoundError:
            module1_result = {
                "status": "unavailable",
                "message": "Module 1 historical scores file is missing."
            }

        result["module1_screening"] = module1_result

        result["module_status"] = {
            "duplicate_work_detection": "completed",
            "financial_anomaly": module2_result["status"],
            "time_cost_overrun": module1_result["status"],
        }

        result["assessment_summary"] = {
            "duplicate_candidates_found": len(result.get("matches", [])),
            "top_duplicate_tier": (
                result["matches"][0]["tier"]
                if result.get("matches")
                else "No candidates returned"
            ),
            "financial_checks": (
                "Module 2 model completed"
                if module2_result["status"] == "completed"
                else "Module 2 unavailable"
            ),
            "time_cost_overrun": (
                # "Historical estimate"
                # if module1_result["status"] == "reference_estimate"
                # else "Unavailable"

                "Live ML prediction"
                if module1_result["status"] == "completed"
                else "Unavailable"
            ),
        }

        result["assessment_notice"] = (
            
                "Time and cost overrun uses SVM models to estimate time-overrun and "
                "under-utilization risk scores for the newly entered work. "
                "Finanicalanomaly detection reference percentiles measure unusualness, not "
                "fraud probability. Duplicate detection ranks potentially similar historical "
                "works; matches require human verification. "
                "No result alone proves fraud or misconduct."
        

        )

        return jsonify(result)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except FileNotFoundError as exc:
        app.logger.exception("Assessment data file missing")
        return jsonify({"error": str(exc)}), 503

    except Exception:
        app.logger.exception("New-work assessment failed")
        return jsonify({
            "error": (
                "Assessment could not be completed. Check the Flask "
                "terminal for the detailed error."
            )
        }), 500



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
