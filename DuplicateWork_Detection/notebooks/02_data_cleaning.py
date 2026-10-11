import pandas as pd
from pathlib import Path

# INPUT = Path(__file__).resolve().parent.parent / "data" / "MPLADS.csv"
# OUTPUT = Path(__file__).resolve().parent.parent / "data" / "mplads_clean.csv"

# df = pd.read_csv(INPUT, sep=";")

# # Remove completely empty rows
# df = df.dropna(how="all").copy()

# # Clean column names
# df.columns = df.columns.str.strip()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = PROJECT_ROOT / "Datasets" / "MPLADS.csv"
OUTPUT = PROJECT_ROOT / "Artifacts" / "module3" / "mplads_clean.csv"

df = pd.read_csv(INPUT, sep=";")
df = df.dropna(how="all").copy()
df.columns = df.columns.str.strip()

# Text columns
text_columns = [
    "MP NAME",
    "WORK",
    "CATEGORY",
    "STATE",
    "CONSTITUENCY",
    "IDA",
    "CITY",
    "WARD",
    "BLOCK",
    "VILLAGE",
    "IDA APPROVAL",
    "STATUS",
    "HOUSE"
]

for col in text_columns:
    df[col] = (
        df[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

# Convert date
df["RECOMMENDED DATE"] = pd.to_datetime(
    df["RECOMMENDED DATE"],
    errors="coerce"
)

# Remove rows without work descriptions
df = df[df["WORK"].str.len() > 0].copy()

# Reset index
df = df.reset_index(drop=True)

# Save
# df.to_csv(OUTPUT, index=False)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("Cleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDate range:")
print(df["RECOMMENDED DATE"].min())
print(df["RECOMMENDED DATE"].max())

print("\nSaved to:")
print(OUTPUT)