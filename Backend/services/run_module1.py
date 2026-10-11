
from pathlib import Path
import sys

# Resolve the project root, regardless of where this script is launched.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Time_Cost_Overrun.predict_all import predict_all


def main():
    output_dir = PROJECT_ROOT / "Artifacts" / "module1"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running Module 1 predictions...")
    print("The model may take some time to train and score the data.")

    predictions = predict_all(seed=42)

    if predictions.empty:
        raise RuntimeError(
            "Module 1 returned no predictions. Check the source datasets."
        )

    output_file = output_dir / "module1_scores.csv"
    predictions.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\nModule 1 prediction run complete")
    print("---------------------------------")
    print(f"Works scored: {len(predictions):,}")
    print(f"Output file: {output_file}")
    print(f"Columns: {list(predictions.columns)}")

    score_columns = [
        column for column in predictions.columns
        if column.endswith("_score")
    ]

    for column in score_columns:
        print(f"\n{column} summary:")
        print(predictions[column].describe().round(3).to_string())

    tier_columns = [
        column for column in predictions.columns
        if column.endswith("_tier")
    ]

    for column in tier_columns:
        print(f"\n{column} distribution:")
        print(predictions[column].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
