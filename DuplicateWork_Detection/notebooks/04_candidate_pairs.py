import pandas as pd
from pathlib import Path

INPUT = Path(__file__).resolve().parent.parent / "data" / "mplads_clean.csv"
OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "candidate_pairs.csv"

df = pd.read_csv(INPUT)

# Replace empty strings with missing values
location_cols = ["STATE", "CONSTITUENCY", "BLOCK", "VILLAGE"]

for col in location_cols:
    df[col] = df[col].replace("", pd.NA)

candidate_pairs = set()


def add_pairs(group):
    """Add all unique pairs from a group."""
    indices = group.index.tolist()

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            candidate_pairs.add((indices[i], indices[j]))


# --------------------------------------------------
# LEVEL 1: STATE + CONSTITUENCY + BLOCK
# --------------------------------------------------

groups = df.dropna(
    subset=["STATE", "CONSTITUENCY", "BLOCK"]
).groupby(
    ["STATE", "CONSTITUENCY", "BLOCK"]
)

for _, group in groups:
    # Avoid extremely large blocks
    if len(group) <= 300:
        add_pairs(group)


# --------------------------------------------------
# LEVEL 2: STATE + CONSTITUENCY + VILLAGE
# --------------------------------------------------

groups = df.dropna(
    subset=["STATE", "CONSTITUENCY", "VILLAGE"]
).groupby(
    ["STATE", "CONSTITUENCY", "VILLAGE"]
)

for _, group in groups:
    if len(group) <= 100:
        add_pairs(group)


# --------------------------------------------------
# LEVEL 3: STATE + CONSTITUENCY
# Only use smaller constituency blocks
# --------------------------------------------------

groups = df.dropna(
    subset=["STATE", "CONSTITUENCY"]
).groupby(
    ["STATE", "CONSTITUENCY"]
)

for _, group in groups:

    # Do not compare huge blocks such as
    # "Sitting Rajya Sabha"
    if len(group) <= 150:
        add_pairs(group)


# --------------------------------------------------
# Create pair dataframe
# --------------------------------------------------

pairs = pd.DataFrame(
    list(candidate_pairs),
    columns=["work_id_1", "work_id_2"]
)

pairs = pairs.sort_values(
    ["work_id_1", "work_id_2"]
).reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

pairs.to_csv(OUTPUT, index=False)

print("Total candidate pairs:", len(pairs))
print("Saved to:", OUTPUT)
