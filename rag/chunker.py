import re


def split_into_sections(text: str) -> list[str]:
    """
    Split document text using numbered section headings
    such as:
    1. Company Overview
    2. Working Hours
    3. Leave Policy
    """

    section_pattern = re.compile(
        r"(?=\b\d+\.\s*[A-Z])"
    )

    sections = section_pattern.split(text)

    return [
        section.strip()
        for section in sections
        if section.strip()
    ]


def split_long_section(
    section: str,
    chunk_size: int,
    chunk_overlap: int
) -> list[str]:
    """
    Split a long section at sentence boundaries.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        section
    )

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if (
            current_chunk
            and len(current_chunk)
            + len(sentence)
            + 1
            <= chunk_size
        ):
            current_chunk += " " + sentence

        else:

            if current_chunk:
                chunks.append(
                    current_chunk.strip()
                )

            current_chunk = sentence

    if current_chunk:
        chunks.append(
            current_chunk.strip()
        )

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[str]:

    if not text or not text.strip():
        return []

    # Normalize line endings
    text = re.sub(
        r"\r\n?",
        "\n",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Split document into numbered sections
    sections = split_into_sections(
        text
    )

    chunks = []

    for section in sections:

        # Keep complete section if it fits
        if len(section) <= chunk_size:

            chunks.append(
                section.strip()
            )

        # Split only very large sections
        else:

            chunks.extend(
                split_long_section(
                    section,
                    chunk_size,
                    chunk_overlap
                )
            )

    return chunks