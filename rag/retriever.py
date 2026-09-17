import faiss
import numpy as np

from rag.pdf_loader import (
    download_pdf,
    extract_text_from_pdf,
)
from rag.chunker import chunk_text
from rag.embeddings import create_embedding, create_embeddings
from rag.vector_store import (
    create_faiss_index,
    save_vector_store,
    load_vector_store,
)


def build_index():
    """
    Read PDFs from Blob Storage,
    chunk them,
    create embeddings,
    and build the FAISS index.
    """

    from azure.storage.blob import BlobServiceClient
    import os

    connection_string = os.environ["AzureWebJobsStorage"]

    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string
    )

    container_client = blob_service_client.get_container_client(
        "pdf-documents"
    )

    all_chunks = []
    metadata = []

    for blob in container_client.list_blobs():

        if not blob.name.lower().endswith(".pdf"):
            continue

        print(f"Processing: {blob.name}")

        pdf_bytes = download_pdf(blob.name)

        text = extract_text_from_pdf(pdf_bytes)

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks):

            all_chunks.append(chunk)

            metadata.append({
                "source": blob.name,
                "chunk_id": chunk_number,
                "text": chunk,
            })

    if not all_chunks:
        raise ValueError(
            "No PDF chunks were found in pdf-documents."
        )

    print(
        f"Creating embeddings for {len(all_chunks)} chunks..."
    )

    embeddings = create_embeddings(all_chunks)

    index = create_faiss_index(embeddings)

    save_vector_store(
        index,
        metadata
    )

    return {
        "documents_processed": len(
            set(item["source"] for item in metadata)
        ),
        "chunks_created": len(all_chunks),
    }


def retrieve(
    query: str,
    top_k: int = 4
):
    """
    Search the FAISS index for relevant chunks.
    """

    index, metadata = load_vector_store()

    query_embedding = create_embedding(query)

    query_vector = np.asarray(
        [query_embedding],
        dtype="float32"
    )

    k = min(
        top_k,
        index.ntotal
    )

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

        item = metadata[index_position].copy()

        item["score"] = float(score)

        results.append(item)

    return results