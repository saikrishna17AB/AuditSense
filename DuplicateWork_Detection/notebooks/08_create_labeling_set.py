import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

FEATURE_FILE = BASE / "outputs" / "training_features.csv"
DATA_FILE = BASE / "data" / "mplads_clean.csv"
OUTPUT_FILE = BASE / "outputs" / "labeling_set.csv"

features = pd.read_csv(FEATURE_FILE)
df = pd.read_csv(DATA_FILE)

# Number of examples from each similarity range
N_PER_GROUP = 100

# Create an overall similarity score ONLY for sampling.
# This is NOT the model and will NOT be used as the final prediction formula.
features["sampling_score"] = (
    0.50 * features["S_text"] +
    0.25 * features["S_geo"] +
    0.15 * features["S_time"] +
    0.10 * features["S_struct"]
)

# Divide pairs into similarity groups
features["similarity_group"] = pd.cut(
    features["sampling_score"],
    bins=[-np.inf, 0.50, 0.65, 0.80, 0.90, np.inf],
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

# Sample from each group
samples = []

for group in features["similarity_group"].cat.categories:

    group_data = features[
        features["similarity_group"] == group
    ]

    n = min(N_PER_GROUP, len(group_data))

    if n > 0:
        samples.append(
            group_data.sample(
                n=n,
                random_state=42
            )
        )

labeling = pd.concat(
    samples,
    ignore_index=True
)

# Shuffle final sample
labeling = labeling.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Add actual work information
work_columns = [
    "WORK",
    "CATEGORY",
    "STATE",
    "CONSTITUENCY",
    "CITY",
    "WARD",
    "BLOCK",
    "VILLAGE",
    "RECOMMENDED DATE"
]

work1 = df[work_columns].copy()
work2 = df[work_columns].copy()

work1.columns = [
    f"WORK_A_{col}" for col in work_columns
]

work2.columns = [
    f"WORK_B_{col}" for col in work_columns
]

labeling = labeling.merge(
    work1,
    left_on="work_id_1",
    right_index=True
)

labeling = labeling.merge(
    work2,
    left_on="work_id_2",
    right_index=True
)

# Empty label column for manual annotation
labeling["LABEL"] = ""

# Remove sampling-only fields from final labeling file
labeling = labeling.drop(
    columns=["sampling_score"],
    errors="ignore"
)

# Put useful columns first
first_columns = [
    "work_id_1",
    "work_id_2",
    "S_text",
    "S_geo",
    "S_time",
    "S_struct",
    "similarity_group",
    "LABEL"
]

remaining_columns = [
    col for col in labeling.columns
    if col not in first_columns
]

labeling = labeling[
    first_columns + remaining_columns
]

labeling.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Labeling dataset created.")
print("Number of pairs:", len(labeling))

print("\nPairs per similarity group:")
print(labeling["similarity_group"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)