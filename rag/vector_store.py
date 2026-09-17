from pathlib import Path
import json

import faiss
import numpy as np


# Store the FAISS index locally during development.
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vector_data"

INDEX_PATH = VECTOR_DIR / "index.faiss"
METADATA_PATH = VECTOR_DIR / "metadata.json"


def save_vector_store(index, metadata):
    VECTOR_DIR.mkdir(exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def load_vector_store():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Vector store does not exist. Build the index first."
        )

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return index, metadata


def create_faiss_index(embeddings):
    vectors = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = vectors.shape[1]

    # Inner product on normalized vectors ≈ cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    return index