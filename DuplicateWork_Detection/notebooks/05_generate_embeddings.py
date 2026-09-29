import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT = Path(__file__).resolve().parent.parent / "data" / "mplads_clean.csv"
OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "sbert_embeddings.npy"

df = pd.read_csv(INPUT)

# Build text for SBERT
def build_text(row):
    parts = [
        str(row["WORK"]),
        f"Category: {row['CATEGORY']}",
        f"State: {row['STATE']}",
        f"Constituency: {row['CONSTITUENCY']}",
        f"City: {row['CITY']}",
        f"Block: {row['BLOCK']}",
        f"Village: {row['VILLAGE']}"
    ]

    return " | ".join(
        part for part in parts
        if part.split(":", 1)[-1].strip()
    )

texts = df.apply(build_text, axis=1).tolist()

print("Number of texts:", len(texts))

# Load pretrained multilingual SBERT
model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(embeddings, dtype=np.float32)

print("Embedding shape:", embeddings.shape)

np.save(OUTPUT, embeddings)

print("Saved embeddings to:")
print(OUTPUT)