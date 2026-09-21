import json
import os
import re
from pathlib import Path

import numpy as np
from azure.storage.blob import BlobServiceClient

from rag.pdf_loader import (
    download_pdf,
    extract_pages_from_pdf
)
from rag.chunker import chunk_text
from rag.embeddings import (
    create_embedding,
    create_embeddings
)
from rag.vector_store import (
    create_faiss_index,
    save_vector_store,
    load_vector_store
)


PDF_CONTAINER_NAME = "pdf-documents"

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vector_data"
MANIFEST_PATH = VECTOR_DIR / "index_manifest.json"

INDEX_VERSION = "v5-section-aware-chunking"


# ============================================================
# INDEX MANIFEST
# ============================================================

def load_index_manifest():
    if not MANIFEST_PATH.exists():
        return None

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_index_manifest(manifest):
    VECTOR_DIR.mkdir(exist_ok=True)

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2
        )


# ============================================================
# BUILD INDEX
# ============================================================

def build_index():

    connection_string = os.environ[
        "AzureWebJobsStorage"
    ]

    blob_service_client = (
        BlobServiceClient.from_connection_string(
            connection_string
        )
    )

    container_client = (
        blob_service_client.get_container_client(
            PDF_CONTAINER_NAME
        )
    )

    pdf_blobs = [
        blob
        for blob in container_client.list_blobs()
        if blob.name.lower().endswith(".pdf")
    ]

    if not pdf_blobs:
        raise ValueError(
            "No PDF files found in the "
            "pdf-documents container."
        )

    current_manifest = {
        blob.name: {
            "etag": blob.etag
        }
        for blob in pdf_blobs
    }

    current_signature = {
        "index_version": INDEX_VERSION,
        "documents": current_manifest
    }

    existing_manifest = load_index_manifest()

    # Reuse existing index when nothing changed
    if existing_manifest == current_signature:

        try:
            _, metadata = load_vector_store()

            return {
                "documents_processed": len(
                    current_manifest
                ),
                "chunks_created": len(
                    metadata
                ),
                "index_rebuilt": False,
                "message": (
                    "Existing FAISS index reused"
                )
            }

        except FileNotFoundError:
            pass

    # --------------------------------------------------------
    # Rebuild index
    # --------------------------------------------------------

    all_chunks = []
    metadata = []

    for blob in pdf_blobs:

        print(
            f"Processing PDF: {blob.name}"
        )

        pdf_bytes = download_pdf(
            blob.name
        )

        # Extract text page by page
        pages = extract_pages_from_pdf(
            pdf_bytes
        )

        # Global chunk ID for this PDF
        chunk_id = 0

        for page in pages:

            chunks = chunk_text(
                page["text"]
            )

            for chunk in chunks:

                all_chunks.append(
                    chunk
                )

                metadata.append({
                    "source": blob.name,
                    "page": page["page"],
                    "chunk_id": chunk_id,
                    "text": chunk
                })

                chunk_id += 1

    if not all_chunks:
        raise ValueError(
            "No text chunks were created."
        )

    print(
        f"Creating embeddings for "
        f"{len(all_chunks)} chunks..."
    )

    embeddings = create_embeddings(
        all_chunks
    )

    index = create_faiss_index(
        embeddings
    )

    save_vector_store(
        index,
        metadata
    )

    save_index_manifest(
        current_signature
    )

    return {
        "documents_processed": len(
            current_manifest
        ),
        "chunks_created": len(
            all_chunks
        ),
        "index_rebuilt": True,
        "message": (
            "FAISS index rebuilt successfully"
        )
    }


# ============================================================
# KEYWORD HELPERS
# ============================================================

def tokenize(text: str) -> set[str]:
    """
    Convert text into lowercase keyword tokens.
    """

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


def keyword_score(
    query: str,
    document: str
) -> float:
    """
    Score based on how many query keywords
    appear in the document chunk.
    """

    query_tokens = tokenize(query)

    if not query_tokens:
        return 0.0

    document_tokens = tokenize(
        document
    )

    matched_tokens = (
        query_tokens & document_tokens
    )

    return (
        len(matched_tokens)
        / len(query_tokens)
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query: str,
    top_k: int = 4,
    min_score: float = 0.35,
    mode: str = "hybrid"
):
    """
    Retrieval modes:

    semantic:
        Uses only semantic similarity.

    hybrid:
        Uses semantic similarity + keyword matching.
    """

    if mode not in ("semantic", "hybrid"):
        raise ValueError(
            "Invalid RAG retrieval mode. "
            "Use 'semantic' or 'hybrid'."
        )

    index, metadata = load_vector_store()

    if index.ntotal <= 0:
        return []

    # --------------------------------------------------------
    # 1. Semantic search
    # --------------------------------------------------------

    query_embedding = create_embedding(
        query
    )

    query_vector = np.asarray(
        [query_embedding],
        dtype="float32"
    )

    candidate_k = min(
        max(top_k * 5, 20),
        index.ntotal
    )

    semantic_scores, indices = index.search(
        query_vector,
        candidate_k
    )

    candidates = []

    # --------------------------------------------------------
    # 2. Calculate final score
    # --------------------------------------------------------

    for semantic_score, index_position in zip(
        semantic_scores[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        item = metadata[
            int(index_position)
        ].copy()

        semantic_score = float(
            semantic_score
        )

        # Convert semantic similarity to 0-1
        semantic_normalized = (
            semantic_score + 1.0
        ) / 2.0

        keyword_match = keyword_score(
            query,
            item["text"]
        )

        # ----------------------------------------------------
        # SEMANTIC MODE
        # ----------------------------------------------------

        if mode == "semantic":
            final_score = semantic_normalized

        # ----------------------------------------------------
        # HYBRID MODE
        # ----------------------------------------------------

        else:
            final_score = (
                0.7 * semantic_normalized
                + 0.3 * keyword_match
            )

        item["semantic_score"] = round(
            semantic_normalized,
            4
        )

        item["keyword_score"] = round(
            keyword_match,
            4
        )

        item["final_score"] = round(
            final_score,
            4
        )

        # Keep "score" for existing response code
        item["score"] = round(
            final_score,
            4
        )

        candidates.append(
            item
        )

    # --------------------------------------------------------
    # 3. Sort by final score
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # 4. Apply minimum score
    # --------------------------------------------------------

    candidates = [
        item
        for item in candidates
        if item["final_score"] >= min_score
    ]

    # --------------------------------------------------------
    # 5. Return final top-K
    # --------------------------------------------------------

    return candidates[:top_k]