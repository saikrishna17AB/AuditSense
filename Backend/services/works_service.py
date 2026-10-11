from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_DIR = PROJECT_ROOT / "Artifacts" / "historical"


def load_index(filename):
    path = HISTORICAL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing index file: {path}. "
            "Run Backend/services/historical_index.py first."
        )

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def load_historical_works():
    """Load sanctioned works and attach completion information when possible."""
    sanctioned = load_index("works_sanctioned_index.csv")
    completed = load_index("works_completed_index.csv")

    sanctioned = sanctioned.copy()
    completed = completed.copy()

    # Use the source serial number as a record identifier.
    sanctioned["source_record_id"] = (
        "SAN-" + sanctioned["Sr. No."].astype(str)
    )
    completed["source_record_id"] = (
        "COMP-" + completed["Sr. No."].astype(str)
    )

    # Normalize column names for the API while retaining original values.
    sanctioned["work_name"] = sanctioned["Work"]
    sanctioned["state"] = sanctioned["State"]
    sanctioned["constituency"] = sanctioned["Constituency"]
    sanctioned["work_category"] = sanctioned["Work category"]
    sanctioned["work_description"] = sanctioned["Work description"]
    sanctioned["sanction_date"] = sanctioned["Sanction Date"]
    sanctioned["sanction_amount"] = sanctioned["Sanction Amount ( ₹ )"]
    sanctioned["work_status"] = sanctioned["Work Status"]
    sanctioned["source_dataset"] = "Works_Sanctioned"

    completed["work_name"] = completed["Work"]
    completed["state"] = completed["State"]
    completed["constituency"] = completed["Constituency"]
    completed["work_category"] = completed["Work Category"]
    completed["work_description"] = completed["Work Description"]
    completed["sanction_date"] = None
    completed["sanction_amount"] = None
    completed["work_status"] = "Completed record"
    completed["source_dataset"] = "Works_Completed"

    # Keep the two source datasets separate for now.
    # Matching them automatically could incorrectly merge different works.
    columns = [
        "source_record_id",
        "source_dataset",
        "work_name",
        "work_category",
        "state",
        "constituency",
        "work_description",
        "sanction_date",
        "sanction_amount",
        "work_status",
    ]

    combined = pd.concat(
        [
            sanctioned.reindex(columns=columns),
            completed.reindex(columns=columns),
        ],
        ignore_index=True,
    )

    return combined


if __name__ == "__main__":
    works = load_historical_works()

    print("Historical works index check")
    print("----------------------------")
    print(f"Total records: {len(works):,}")
    print("\nRecords by source:")
    print(works["source_dataset"].value_counts().to_string())
    print("\nSample records:")
    print(
        works[
            [
                "source_record_id",
                "work_name",
                "state",
                "work_status",
            ]
        ].head(5).to_string(index=False)
    )

    output_path = HISTORICAL_DIR / "historical_works_index.csv"
    works.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {output_path}")