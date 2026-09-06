from .base import BaseChunker
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RecursiveChunker(BaseChunker):

    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, texts):
        logger.info(f"Chunking {len(texts)} document(s) (size={self.chunk_size}, overlap={self.overlap})")
        chunks = []
        chunk_id = 1

        for text in texts:
            for part in self._split(text):
                chunks.append({
                    "id": chunk_id,
                    "text": part
                })
                chunk_id += 1

        logger.info(f"Generated total of {len(chunks)} chunks")
        return chunks

    def _split(self, text):
        if len(text) <= self.chunk_size:
            return [text]

        separators = ["\n\n", "\n", ". ", " "]

        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                chunks = []
                current = ""

                for part in parts:
                    if len(current) + len(part) + len(separator) <= self.chunk_size:
                        current += part + separator
                    else:
                        if current:
                            chunks.append(current.strip())

                        current = (
                            current[-self.overlap:] + part + separator
                            if self.overlap
                            else part + separator
                        )

                if current:
                    chunks.append(current.strip())

                return chunks

        return [
            text[i:i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.overlap)
        ]