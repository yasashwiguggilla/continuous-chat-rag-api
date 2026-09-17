import os

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


# ============================================================
# BUILD FAISS INDEX
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

    all_chunks = []
    metadata = []

    for blob in container_client.list_blobs():

        if not blob.name.lower().endswith(".pdf"):
            continue

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
            "No PDF chunks found in "
            "the pdf-documents container."
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

    documents_processed = len(
        set(
            item["source"]
            for item in metadata
        )
    )

    return {
        "documents_processed": (
            documents_processed
        ),
        "chunks_created": len(
            all_chunks
        )
    }


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve(
    query: str,
    top_k: int = 4,
    min_score: float = 0.35
):
    """
    Retrieve the most relevant PDF chunks.

    top_k:
        Maximum number of chunks to retrieve.

    min_score:
        Minimum similarity score required
        for a chunk to be returned.
    """

    index, metadata = load_vector_store()

    # Create embedding for user's question
    query_embedding = create_embedding(
        query
    )

    query_vector = np.asarray(
        [query_embedding],
        dtype="float32"
    )

    # Don't ask FAISS for more vectors
    # than actually exist.
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

        # Ignore weak matches
        if score < min_score:
            continue

        item = metadata[
            int(index_position)
        ].copy()

        item["score"] = score

        results.append(item)

    return results