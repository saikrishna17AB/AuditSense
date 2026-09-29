import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

PAIR_FILE = BASE / "outputs" / "candidate_pairs.csv"
EMBEDDING_FILE = BASE / "outputs" / "sbert_embeddings.npy"
OUTPUT_FILE = BASE / "outputs" / "semantic_pairs.csv"

# Load candidate pairs
pairs = pd.read_csv(PAIR_FILE)

# Load normalized SBERT embeddings
embeddings = np.load(EMBEDDING_FILE)

print("Candidate pairs:", len(pairs))
print("Embedding shape:", embeddings.shape)

# Convert pair IDs to integer arrays
idx1 = pairs["work_id_1"].to_numpy(dtype=np.int64)
idx2 = pairs["work_id_2"].to_numpy(dtype=np.int64)

# Calculate cosine similarity using dot product
semantic_similarity = np.sum(
    embeddings[idx1] * embeddings[idx2],
    axis=1
)

# Add feature
pairs["S_text"] = semantic_similarity

# Save
pairs.to_csv(OUTPUT_FILE, index=False)

print("\nSemantic similarity calculated.")
print("Output shape:", pairs.shape)
print("\nS_text statistics:")
print(pairs["S_text"].describe())

print("\nSaved to:")
print(OUTPUT_FILE)