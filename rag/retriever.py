import json
import os
from pathlib import Path

import numpy as np
from azure.storage.blob import BlobServiceClient

from rag.pdf_loader import (
    download_pdf,
    extract_text_from_pdf
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

# Change this whenever your chunking/embedding logic changes.
INDEX_VERSION = "v2-paragraph-chunking"


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

    existing_manifest = load_index_manifest()

    current_signature = {
        "index_version": INDEX_VERSION,
        "documents": current_manifest
    }

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

        text = extract_text_from_pdf(
            pdf_bytes
        )

        chunks = chunk_text(
            text
        )

        for chunk_id, chunk in enumerate(
            chunks
        ):

            all_chunks.append(chunk)

            metadata.append({
                "source": blob.name,
                "chunk_id": chunk_id,
                "text": chunk
            })

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


def retrieve(
    query: str,
    top_k: int = 4,
    min_score: float = 0.35
):
    """
    Retrieve the most relevant PDF chunks.
    """

    index, metadata = load_vector_store()

    query_embedding = create_embedding(
        query
    )

    query_vector = np.asarray(
        [query_embedding],
        dtype="float32"
    )

    k = min(
        top_k,
        index.ntotal
    )

    if k <= 0:
        return []

    scores, indices = index.search(
        query_vector,
        k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        score = float(score)

        if score < min_score:
            continue

        item = metadata[
            int(index_position)
        ].copy()

        item["score"] = score

        results.append(item)

    return results