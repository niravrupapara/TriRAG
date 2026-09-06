from .base import BaseChunker


class SimpleChunker(BaseChunker):

    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents):
        chunks = []
        chunk_id = 1

        for doc in documents:
            start = 0

            while start < len(doc):
                end = start + self.chunk_size
                text = " ".join(doc[start:end].split())

                chunks.append({
                    "id": chunk_id,
                    "text": text
                })

                chunk_id += 1
                start = end - self.overlap

        return chunks