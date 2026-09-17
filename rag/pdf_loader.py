import os
from io import BytesIO

from azure.storage.blob import BlobServiceClient
from pypdf import PdfReader


PDF_CONTAINER_NAME = "pdf-documents"


def download_pdf(blob_name: str) -> bytes:
    connection_string = os.environ["AzureWebJobsStorage"]

    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string
    )

    container_client = blob_service_client.get_container_client(
        PDF_CONTAINER_NAME
    )

    blob_client = container_client.get_blob_client(blob_name)

    return blob_client.download_blob().readall()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)