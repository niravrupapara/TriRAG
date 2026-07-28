from typing import List
from src.utils.logging import get_logger

logger = get_logger(__name__)

def fixed_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into fixed-size chunks with overlap."""

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    logger.info(f"Text chunked into {len(chunks)} chunks | chunk size: {chunk_size} | overlap: {overlap}")
    return chunks

def recursive_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text Recursively by paragraph, sentence, and word until it fits the chunk size."""
    separators = ['\n\n', '\n', '. ', ' ']

    def split(text, separators):
        if len(text) <= chunk_size:
            return [text]

        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current = ""
                for part in parts:
                    if len(current) + len(part) + len(sep) <= chunk_size:
                        current += part + sep
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = current[-overlap:] + part + sep if overlap else part + sep
                if current:
                    chunks.append(current.strip())
                return chunks
        return fixed_chunk(text, chunk_size, overlap)

    chunks = split(text, separators)
    logger.info(f"Text recursively chunked into {len(chunks)} chunks | chunk size: {chunk_size} | overlap: {overlap}")
    return chunks


def chunk_text(text: str, strategy: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text using the specified strategy and return list of chunks."""
    logger.debug(f"Chunking text | strategy: {strategy} | chunk size: {chunk_size} | overlap: {overlap}")
    if strategy == 'fixed':
        return fixed_chunk(text, chunk_size, overlap)
    elif strategy == 'recursive':
        return recursive_chunk(text, chunk_size, overlap)
    else:
        logger.error(f"Unsupported chunking strategy: {strategy}")
        raise ValueError(f"Unsupported chunking strategy: {strategy}")