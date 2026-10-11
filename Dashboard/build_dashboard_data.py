
from pathlib import Path
import json
import math

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT / "Artifacts"
DATASET = PROJECT / "Datasets" / "MPLADS.csv"
DATA_OUT = Path(__file__).resolve().parent / "data"

M1_PATH = ARTIFACTS / "module1" / "module1_scores.csv"
M2_PATH = ARTIFACTS / "module2" / "module2_scores.csv"
M3_PATH = ARTIFACTS / "module3" / "duplicate_scores.csv"


def load_csv(path, sep=","):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, sep=sep, low_memory=False)


def clean_value(value):
    """Convert pandas/numpy values into JSON-safe Python values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def records(df, limit=None):
    """Convert a DataFrame to JSON-safe dictionaries."""
    if limit is not None:
        df = df.head(limit)

    return [
        {str(k): clean_value(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def first_existing(df, candidates, default=None):
    """Return the first available column from a list of names."""
    for name in candidates:
        if name in df.columns:
            return name
    return default


def count_values(series, expected):
    values = series.fillna("").astype(str)
    return {
        label: int(values.eq(label).sum())
        for label in expected
    }


def main():
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    # Read the actual model artifacts.
    m1 = load_csv(M1_PATH)
    m2 = load_csv(M2_PATH)
    m3 = load_csv(M3_PATH)

    # -----------------------------------------------------
    # Module 1: time overrun and underutilization
    # -----------------------------------------------------
    time_tier_col = first_existing(
        m1, ["time_overrun_tier", "time_tier"]
    )
    under_tier_col = first_existing(
        m1, ["underutilization_tier", "underutil_tier"]
    )

    m1_summary = {
        "works_scored": int(len(m1)),
        "time_overrun_tiers": (
            count_values(m1[time_tier_col], ["Low", "Medium", "High"])
            if time_tier_col else {}
        ),
        "underutilization_tiers": (
            count_values(
                m1[under_tier_col], ["Low", "Medium", "High"]
            )
            if under_tier_col else {}
        ),
    }

    # -----------------------------------------------------
    # Module 2: financial anomaly detection
    # -----------------------------------------------------
    m2_tier_col = first_existing(
        m2, ["agreement_tier", "m2_tier", "financial_tier"]
    )
    fused_col = first_existing(
        m2, ["fused_score", "m2_fused_score"]
    )

    m2_summary = {
        "works_scored": int(len(m2)),
        "tier_counts": (
            {
                str(k): int(v)
                for k, v in m2[m2_tier_col]
                .fillna("Unknown").astype(str)
                .value_counts().items()
            }
            if m2_tier_col else {}
        ),
        "mean_fused_score": (
            round(float(pd.to_numeric(
                m2[fused_col], errors="coerce"
            ).mean()), 2)
            if fused_col else None
        ),
    }

    # -----------------------------------------------------
    # Module 3: pair-level duplicate scoring
    # -----------------------------------------------------
    score_col = first_existing(
        m3, ["duplicate_score", "score"]
    )
    result_col = first_existing(
        m3, ["result", "duplicate_band", "band"]
    )

    m3_summary = {
        "candidate_pairs_scored": int(len(m3)),
        "score_mean": (
            round(float(pd.to_numeric(
                m3[score_col], errors="coerce"
            ).mean()), 4)
            if score_col else None
        ),
        "band_counts": (
            {
                str(k): int(v)
                for k, v in m3[result_col]
                .fillna("Unknown").astype(str)
                .value_counts().items()
            }
            if result_col else {}
        ),
    }

    # -----------------------------------------------------
    # Enrich Module 3 pair indices with original works.
    # Pair IDs are row indices from the cleaned MPLADS data.
    # -----------------------------------------------------
    source = load_csv(DATASET, sep=";")
    source.columns = [str(c).strip().strip('"') for c in source.columns]
    source = source.dropna(how="all").reset_index(drop=True)

    # Normalize common fields from the source dataset.
    work_col = first_existing(source, ["WORK", "Work", "work"])
    state_col = first_existing(source, ["STATE", "State", "state"])
    category_col = first_existing(
        source, ["CATEGORY", "Category", "category"]
    )
    village_col = first_existing(
        source, ["VILLAGE", "Village", "village"]
    )
    block_col = first_existing(
        source, ["BLOCK", "Block", "block"]
    )
    amount_col = first_existing(
        source, ["ALLOCATION AMOUNT", "Allocation Amount", "amount"]
    )

    def work_at(index):
        try:
            idx = int(index)
            if 0 <= idx < len(source):
                row = source.iloc[idx]
                return {
                    "work_id": idx,
                    "description": clean_value(
                        row[work_col] if work_col else None
                    ),
                    "state": clean_value(
                        row[state_col] if state_col else None
                    ),
                    "category": clean_value(
                        row[category_col] if category_col else None
                    ),
                    "village": clean_value(
                        row[village_col] if village_col else None
                    ),
                    "block": clean_value(
                        row[block_col] if block_col else None
                    ),
                    "amount": clean_value(
                        row[amount_col] if amount_col else None
                    ),
                }
        except (TypeError, ValueError, IndexError):
            pass
        return None

    pair_a_col = first_existing(m3, ["work_id_1", "id_1"])
    pair_b_col = first_existing(m3, ["work_id_2", "id_2"])

    enriched_pairs = []
    if pair_a_col and pair_b_col:
        for row in m3.to_dict(orient="records"):
            a = work_at(row.get(pair_a_col))
            b = work_at(row.get(pair_b_col))

            enriched_pairs.append({
                "work_a": a,
                "work_b": b,
                "duplicate_score": clean_value(
                    row.get(score_col) if score_col else None
                ),
                "result": clean_value(
                    row.get(result_col) if result_col else None
                ),
                "S_text": clean_value(row.get("S_text")),
                "S_geo": clean_value(row.get("S_geo")),
                "S_time": clean_value(row.get("S_time")),
                "S_struct": clean_value(row.get("S_struct")),
            })

    # Work lists are independent. No row-position join between models.
    # Limit browser payload size while retaining full summary counts.
    # work_records = {
    #     "module1": records(m1, limit=500),
    #     "module2": records(m2, limit=500),
    # }

    
    # Work lists are independent. Never join model outputs by row position.
    # Keep full counts in the KPI summary, but send the highest-ranked
    # Module 2 records to the browser for historical review.
    work_records = {
        "module1": records(m1, limit=500),
        "module2": [],
    }

    m2_display = m2.copy()

    # fused_score is used only to prioritize records for review.
    # The UI should display component scores and agreement tier separately.
    if "fused_score" in m2_display.columns:
        m2_display["_review_rank"] = pd.to_numeric(
            m2_display["fused_score"], errors="coerce"
        )
        m2_display = m2_display.sort_values(
            "_review_rank", ascending=False, na_position="last"
        ).drop(columns=["_review_rank"])

    # Preserve available raw fields, model scores, rule flags,
    # and evidence features in the JSON output.
    evidence_columns = [
        "sr_no",
        "work_raw",
        "work_description",
        "work_category",
        "state",
        "mp_name",
        "constituency",
        "agency_raw",
        "amount",
        "recommended_date",
        "sanction_date",
        "work_status",
        "isolation_forest_score",
        "autoencoder_score",
        "fused_score",
        "agreement_tier",
        "flag_threshold_avoidance",
        "flag_fast_approval",
        "flag_pricing_outlier",
        "f1_threshold_avoidance",
        "f2_benford_excess",
        "f3_round_number",
        "f4_agency_cost_trend",
        "f5_approval_gap",
        "f6_agency_fingerprint",
        "top_contributing_factors",
    ]

    available_evidence_columns = [
        col for col in evidence_columns if col in m2_display.columns
    ]

    work_records["module2"] = records(
        m2_display[available_evidence_columns].head(500)
    )

    # Sort the sample by score for useful review-first presentation.
    if score_col and not m3.empty:
        m3_sorted = m3.copy()
        m3_sorted[score_col] = pd.to_numeric(
            m3_sorted[score_col], errors="coerce"
        )
        m3_sorted = m3_sorted.sort_values(
            score_col, ascending=False, na_position="last"
        )
        sort_ids = m3_sorted.index.tolist()
        pair_by_index = {
            i: enriched_pairs[i] for i in range(len(enriched_pairs))
        }
        pair_sample = [
            pair_by_index[i] for i in sort_ids[:500]
            if i in pair_by_index
        ]
    else:
        pair_sample = enriched_pairs[:500]

    payloads = {
        "kpi_summary.json": {
            "generated_from": "Artifacts CSV files",
            "module1": m1_summary,
            "module2": m2_summary,
            "module3": m3_summary,
        },
        "module1_data.json": work_records["module1"],
        "module2_data.json": work_records["module2"],
        "module3_pairs.json": pair_sample,
    }

    for filename, payload in payloads.items():
        destination = DATA_OUT / filename
        with destination.open("w", encoding="utf-8") as f:
            json.dump(
                payload, f, indent=2, ensure_ascii=False,
                allow_nan=False
            )
        print(f"Wrote {destination} ({destination.stat().st_size:,} bytes)")

    print("\n=== Dashboard data build complete ===")
    print(f"Module 1 works: {len(m1):,}")
    print(f"Module 2 works: {len(m2):,}")
    print(f"Module 3 candidate pairs: {len(m3):,}")
    print(f"Module 3 pairs enriched: {len(enriched_pairs):,}")
    print(f"Module 3 browser sample: {len(pair_sample):,}")


if __name__ == "__main__":
    main()
