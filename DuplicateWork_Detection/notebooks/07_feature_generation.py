import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz.fuzz import ratio

BASE = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE / "outputs" / "semantic_pairs.csv"
DATA_FILE = BASE / "data" / "mplads_clean.csv"
OUTPUT_FILE = BASE / "outputs" / "training_features.csv"

pairs = pd.read_csv(INPUT_FILE)
data = pd.read_csv(DATA_FILE)

print(f"Calculating features for {len(pairs):,} pairs...")

# ---------------------------------------------------------
# Prepare original records
# ---------------------------------------------------------

records = data[
    [
        "STATE",
        "CONSTITUENCY",
        "CITY",
        "WARD",
        "BLOCK",
        "VILLAGE",
        "RECOMMENDED DATE",
        "CATEGORY",
        "IDA APPROVAL",
        "HOUSE"
    ]
].copy()


def clean_value(value):
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


# ---------------------------------------------------------
# Location similarity
# ---------------------------------------------------------

def field_similarity(a, b):
    a = clean_value(a)
    b = clean_value(b)

    if not a or not b:
        return None

    return 1.0 if a == b else 0.0


def village_similarity(a, b):
    a = clean_value(a)
    b = clean_value(b)

    if not a or not b:
        return None

    return ratio(a, b) / 100.0


def calculate_geo_similarity(row_a, row_b):

    similarities = []
    weights = []

    # Broad administrative level
    state = field_similarity(
        row_a["STATE"],
        row_b["STATE"]
    )

    if state is not None:
        similarities.append(state)
        weights.append(0.10)

    # Constituency
    constituency = field_similarity(
        row_a["CONSTITUENCY"],
        row_b["CONSTITUENCY"]
    )

    if constituency is not None:
        similarities.append(constituency)
        weights.append(0.15)

    # City
    city = field_similarity(
        row_a["CITY"],
        row_b["CITY"]
    )

    if city is not None:
        similarities.append(city)
        weights.append(0.10)

    # Ward
    ward = field_similarity(
        row_a["WARD"],
        row_b["WARD"]
    )

    if ward is not None:
        similarities.append(ward)
        weights.append(0.10)

    # Block
    block = field_similarity(
        row_a["BLOCK"],
        row_b["BLOCK"]
    )

    if block is not None:
        similarities.append(block)
        weights.append(0.20)

    # Village - fuzzy matching
    village = village_similarity(
        row_a["VILLAGE"],
        row_b["VILLAGE"]
    )

    if village is not None:
        similarities.append(village)
        weights.append(0.35)

    if not weights:
        return 0.0

    return np.average(similarities, weights=weights)


# ---------------------------------------------------------
# Temporal similarity
# ---------------------------------------------------------

def calculate_time_similarity(date_a, date_b):

    date_a = pd.to_datetime(date_a)
    date_b = pd.to_datetime(date_b)

    delta_days = abs((date_a - date_b).days)

    return np.exp(-delta_days / 180.0)


# ---------------------------------------------------------
# Structured similarity
# ---------------------------------------------------------

def calculate_structured_similarity(row_a, row_b):

    fields = [
        "CATEGORY",
        "IDA APPROVAL",
        "HOUSE"
    ]

    similarities = []

    for field in fields:

        a = clean_value(row_a[field])
        b = clean_value(row_b[field])

        if a and b:
            similarities.append(
                1.0 if a == b else 0.0
            )

    if not similarities:
        return 0.0

    return np.mean(similarities)


# ---------------------------------------------------------
# Calculate features
# ---------------------------------------------------------

S_geo = []
S_time = []
S_struct = []

for count, (_, pair) in enumerate(pairs.iterrows()):

    id_a = int(pair["work_id_1"])
    id_b = int(pair["work_id_2"])

    row_a = records.iloc[id_a]
    row_b = records.iloc[id_b]

    # Geographic similarity
    geo = calculate_geo_similarity(
        row_a,
        row_b
    )

    # Temporal similarity
    time = calculate_time_similarity(
        row_a["RECOMMENDED DATE"],
        row_b["RECOMMENDED DATE"]
    )

    # Structured similarity
    struct = calculate_structured_similarity(
        row_a,
        row_b
    )

    S_geo.append(geo)
    S_time.append(time)
    S_struct.append(struct)

    if (count + 1) % 100000 == 0:
        print(
            f"Processed {count + 1:,} / "
            f"{len(pairs):,}"
        )


# ---------------------------------------------------------
# Add features
# ---------------------------------------------------------

pairs["S_geo"] = S_geo
pairs["S_time"] = S_time
pairs["S_struct"] = S_struct


print("\nFeature generation complete.")

print("\nFeature statistics:")
print(
    pairs[
        [
            "S_text",
            "S_geo",
            "S_time",
            "S_struct"
        ]
    ].describe()
)


pairs.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)