import re


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[str]:
    """
    Create paragraph-aware chunks.

    The function tries to keep complete paragraphs together
    while respecting the maximum chunk size.
    """

    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Split on blank lines so paragraphs are preserved.
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        # If adding this paragraph still fits,
        # keep it in the current chunk.
        if (
            current_chunk
            and len(current_chunk) + len(paragraph) + 1
            <= chunk_size
        ):
            current_chunk += "\n\n" + paragraph
            continue

        # Save the existing chunk.
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If one paragraph itself is too large,
        # split it safely.
        if len(paragraph) > chunk_size:

            start = 0

            while start < len(paragraph):

                end = start + chunk_size

                piece = paragraph[start:end].strip()

                if piece:
                    chunks.append(piece)

                start += chunk_size - chunk_overlap

            current_chunk = ""

        else:
            current_chunk = paragraph

    # Add the final chunk.
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks