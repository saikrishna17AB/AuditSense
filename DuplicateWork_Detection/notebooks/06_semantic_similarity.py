# import pandas as pd
# import numpy as np
# from pathlib import Path

# BASE = Path(__file__).resolve().parent.parent

# # PAIR_FILE = BASE / "outputs" / "candidate_pairs.csv"
# # EMBEDDING_FILE = BASE / "outputs" / "sbert_embeddings.npy"
# # OUTPUT_FILE = BASE / "outputs" / "semantic_pairs.csv"



# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# MODULE3_DIR = PROJECT_ROOT / "Artifacts" / "module3"

# PAIR_FILE = MODULE3_DIR / "candidate_pairs.csv"
# EMBEDDING_FILE = MODULE3_DIR / "sbert_embeddings.npy"
# OUTPUT_FILE = MODULE3_DIR / "semantic_pairs.csv"


# # Load candidate pairs
# pairs = pd.read_csv(PAIR_FILE)

# # Load normalized SBERT embeddings
# embeddings = np.load(EMBEDDING_FILE)

# print("Candidate pairs:", len(pairs))
# print("Embedding shape:", embeddings.shape)

# # Convert pair IDs to integer arrays
# idx1 = pairs["work_id_1"].to_numpy(dtype=np.int64)
# idx2 = pairs["work_id_2"].to_numpy(dtype=np.int64)

# # Calculate cosine similarity using dot product
# semantic_similarity = np.sum(
#     embeddings[idx1] * embeddings[idx2],
#     axis=1
# )

# # Add feature
# pairs["S_text"] = semantic_similarity

# # Save
# pairs.to_csv(OUTPUT_FILE, index=False)

# print("\nSemantic similarity calculated.")
# print("Output shape:", pairs.shape)
# print("\nS_text statistics:")
# print(pairs["S_text"].describe())

# print("\nSaved to:")
# print(OUTPUT_FILE)




import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE3_DIR = PROJECT_ROOT / "Artifacts" / "module3"

PAIR_FILE = MODULE3_DIR / "candidate_pairs.csv"
EMBEDDING_FILE = MODULE3_DIR / "sbert_embeddings.npy"
OUTPUT_FILE = MODULE3_DIR / "semantic_pairs.csv"

# Load candidate pairs and the existing embeddings
pairs = pd.read_csv(PAIR_FILE)
embeddings = np.load(EMBEDDING_FILE, mmap_mode="r")

print("Candidate pairs:", len(pairs))
print("Embedding shape:", embeddings.shape)

idx1 = pairs["work_id_1"].to_numpy(dtype=np.int64)
idx2 = pairs["work_id_2"].to_numpy(dtype=np.int64)

# Process small batches to limit RAM usage
semantic_similarity = np.empty(len(pairs), dtype=np.float32)
batch_size = 5000

for start in range(0, len(pairs), batch_size):
    end = min(start + batch_size, len(pairs))

    emb1 = embeddings[idx1[start:end]]
    emb2 = embeddings[idx2[start:end]]

    # Embeddings are normalized, so dot product = cosine similarity
    semantic_similarity[start:end] = np.einsum(
        "ij,ij->i", emb1, emb2
    )

    if end % 50000 == 0 or end == len(pairs):
        print(f"Processed {end:,} / {len(pairs):,} pairs")

pairs["S_text"] = semantic_similarity

pairs.to_csv(OUTPUT_FILE, index=False)

print("\nSemantic similarity calculated.")
print("\nS_text statistics:")
print(pairs["S_text"].describe())
print("\nSaved to:", OUTPUT_FILE)
