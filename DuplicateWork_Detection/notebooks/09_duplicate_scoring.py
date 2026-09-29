import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE / "outputs" / "training_features.csv"
OUTPUT_FILE = BASE / "outputs" / "duplicate_scores.csv"

df = pd.read_csv(INPUT_FILE)

# Weighted duplicate score
df["duplicate_score"] = (
    0.50 * df["S_text"]
    + 0.20 * df["S_geo"]
    + 0.15 * df["S_time"]
    + 0.15 * df["S_struct"]
)

# Classification based on documented thresholds
def classify(score):
    if score >= 0.75:
        return "Potential Duplicate"
    elif score >= 0.55:
        return "Needs Review"
    else:
        return "Lower Likelihood"

df["result"] = df["duplicate_score"].apply(classify)

# Sort highest-risk pairs first
df = df.sort_values(
    "duplicate_score",
    ascending=False
).reset_index(drop=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Duplicate scoring complete.")

print("\nTotal candidate pairs:", len(df))

print("\nResult distribution:")
print(df["result"].value_counts())

print("\nDuplicate score statistics:")
print(df["duplicate_score"].describe())

print("\nTop 20 potential duplicates:")
print(
    df[
        [
            "work_id_1",
            "work_id_2",
            "S_text",
            "S_geo",
            "S_time",
            "S_struct",
            "duplicate_score",
            "result"
        ]
    ].head(20).to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)