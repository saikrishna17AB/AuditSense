import pandas as pd
from pathlib import Path

INPUT = Path(__file__).resolve().parent.parent / "data" / "mplads_clean.csv"

df = pd.read_csv(INPUT)

# Convert empty strings to missing values for analysis
location_cols = ["STATE", "CONSTITUENCY", "CITY", "WARD", "BLOCK", "VILLAGE"]

for col in location_cols:
    df[col] = df[col].replace("", pd.NA)

print("Total records:", len(df))

print("\nRecords by STATE:")
print(df["STATE"].value_counts().head(20))

print("\nUnique values:")
for col in location_cols:
    print(f"{col}: {df[col].nunique(dropna=True)} unique")

print("\nMissing values:")
print(df[location_cols].isna().sum())

# Primary blocking key
df["STATE_CONSTITUENCY"] = (
    df["STATE"].fillna("UNKNOWN") + "||" +
    df["CONSTITUENCY"].fillna("UNKNOWN")
)

group_sizes = df["STATE_CONSTITUENCY"].value_counts()

print("\nSTATE + CONSTITUENCY blocks:")
print("Number of blocks:", len(group_sizes))
print("Largest block:", group_sizes.max())
print("Average block size:", group_sizes.mean())
print("Median block size:", group_sizes.median())

print("\nLargest 20 blocks:")
print(group_sizes.head(20))