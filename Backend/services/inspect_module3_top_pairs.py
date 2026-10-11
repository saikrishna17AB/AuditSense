
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE3_DIR = PROJECT_ROOT / "Artifacts" / "module3"

SCORES_FILE = MODULE3_DIR / "duplicate_scores.csv"
WORKS_FILE = MODULE3_DIR / "mplads_clean.csv"
OUTPUT_FILE = MODULE3_DIR / "top_pairs_for_review.csv"

scores = pd.read_csv(SCORES_FILE)
works = pd.read_csv(WORKS_FILE, sep=None, engine="python")

# Inspect the highest-scoring pairs first.
top = scores.sort_values(
    "duplicate_score", ascending=False
).head(200).copy()

works = works.reset_index(drop=True)
top["work_id_1"] = top["work_id_1"].astype(int)
top["work_id_2"] = top["work_id_2"].astype(int)

# Attach original work details to both sides of each pair.
details = [
    "MP NAME", "WORK", "STATE", "CONSTITUENCY", "CITY",
    "WARD", "BLOCK", "VILLAGE", "RECOMMENDED DATE",
    "CATEGORY", "IDA APPROVAL", "HOUSE", "ALLOCATION AMOUNT"
]

available = [c for c in details if c in works.columns]

left = works[available].add_prefix("work1_")
right = works[available].add_prefix("work2_")

top = top.join(
    left.reindex(top["work_id_1"].to_numpy()).reset_index(drop=True)
)
top = top.join(
    right.reindex(top["work_id_2"].to_numpy()).reset_index(drop=True)
)

top.to_csv(OUTPUT_FILE, index=False)

print("Top-pair review file saved:", OUTPUT_FILE)
print("Pairs exported:", len(top))
print("\nReview columns:")
print(top.columns.tolist())
print("\nTop 10 pairs with original work descriptions:")
for _, row in top.head(10).iterrows():
    print("\n" + "=" * 70)
    print("IDs:", row["work_id_1"], "vs", row["work_id_2"])
    print("Score:", round(row["duplicate_score"], 3))
    print("Work 1:", row.get("work1_WORK", ""))
    print("Work 2:", row.get("work2_WORK", ""))
    print("State:", row.get("work1_STATE", ""), "|",
          row.get("work2_STATE", ""))
    print("Constituency:", row.get("work1_CONSTITUENCY", ""), "|",
          row.get("work2_CONSTITUENCY", ""))
    print("Village:", row.get("work1_VILLAGE", ""), "|",
          row.get("work2_VILLAGE", ""))
