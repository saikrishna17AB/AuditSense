import pandas as pd
from pathlib import Path

file_path = Path(__file__).resolve().parent.parent / "data" / "MPLADS.csv"

df = pd.read_csv(file_path, sep=";")

print("Dataset shape:", df.shape)

print("\nColumns:")
for col in df.columns:
    print(col)

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)