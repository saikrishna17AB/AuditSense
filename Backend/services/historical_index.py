
from pathlib import Path
import pandas as pd

# Project root: MPLADS_Solution_Code/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Datasets"
OUTPUT_DIR = PROJECT_ROOT / "Artifacts" / "historical"

SANCTIONED_FILE = DATA_DIR / "Works_Sanctioned.csv"
COMPLETED_FILE = DATA_DIR / "Works Completed.csv"
MPLADS_FILE = DATA_DIR / "MPLADS.csv"


def read_csv_safely(path: Path, sep=None) -> pd.DataFrame:
    """Read a CSV while preserving its original column names."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if sep is not None:
        return pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)

    # The works datasets may use comma separation; detect if necessary.
    return pd.read_csv(path, sep=None, engine="python",
                       encoding="utf-8-sig")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(column).strip().replace("\ufeff", "")
        for column in df.columns
    ]
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sanctioned = clean_columns(read_csv_safely(SANCTIONED_FILE))
    completed = clean_columns(read_csv_safely(COMPLETED_FILE))
    mplads = clean_columns(read_csv_safely(MPLADS_FILE, sep=";"))

    print("Dataset loading check")
    print("---------------------")
    for name, df in [
        ("Works sanctioned", sanctioned),
        ("Works completed", completed),
        ("MPLADS", mplads),
    ]:
        print(f"{name}: {len(df):,} rows")
        print(f"  Columns: {list(df.columns)}")

    # Preserve the source records independently. We will add stable IDs
    # after checking the real identifier columns in each dataset.
    sanctioned.to_csv(
        OUTPUT_DIR / "works_sanctioned_index.csv", index=False
    )
    completed.to_csv(
        OUTPUT_DIR / "works_completed_index.csv", index=False
    )
    mplads.to_csv(
        OUTPUT_DIR / "mplads_index.csv", index=False
    )

    print("\nSaved source indexes to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
