import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

SCORES = BASE / "outputs" / "duplicate_scores.csv"
DATA = BASE / "data" / "mplads_clean.csv"
OUTPUT = BASE / "outputs" / "village_mismatch_inspection.csv"

scores = pd.read_csv(SCORES)
data = pd.read_csv(DATA)

# Add original record information
cols = [
    "WORK",
    "STATE",
    "CONSTITUENCY",
    "CITY",
    "WARD",
    "BLOCK",
    "VILLAGE",
    "RECOMMENDED DATE"
]

a = data[cols].copy()
b = data[cols].copy()

a.columns = [f"A_{c}" for c in cols]
b.columns = [f"B_{c}" for c in cols]

scores = scores.merge(
    a,
    left_on="work_id_1",
    right_index=True
)

scores = scores.merge(
    b,
    left_on="work_id_2",
    right_index=True
)

# Normalize village names for comparison
village_a = scores["A_VILLAGE"].fillna("").astype(str).str.strip().str.lower()
village_b = scores["B_VILLAGE"].fillna("").astype(str).str.strip().str.lower()

# Keep pairs where both villages exist and are different
mask = (
    (village_a != "") &
    (village_b != "") &
    (village_a != village_b) &
    (scores["duplicate_score"] >= 0.75)
)

result = scores.loc[mask].copy()

# Highest scores first
result = result.sort_values(
    "duplicate_score",
    ascending=False
)

# Keep the most useful columns
result = result[
    [
        "work_id_1",
        "work_id_2",
        "A_WORK",
        "B_WORK",
        "A_STATE",
        "B_STATE",
        "A_CONSTITUENCY",
        "B_CONSTITUENCY",
        "A_BLOCK",
        "B_BLOCK",
        "A_VILLAGE",
        "B_VILLAGE",
        "S_text",
        "S_geo",
        "S_time",
        "S_struct",
        "duplicate_score",
        "result"
    ]
]

result.to_csv(
    OUTPUT,
    index=False
)

print("High-scoring pairs with different villages:", len(result))

print("\nFirst 30 examples:")
print(result.head(30).to_string(index=False))

print("\nSaved to:")
print(OUTPUT)