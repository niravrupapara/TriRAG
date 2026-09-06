from abc import ABC, abstractmethod


class BaseVectorStore(ABC):

    @abstractmethod
    def add(self, chunks, embeddings):
        pass

    @abstractmethod
    def search(self, query_embedding, k=5):
        pass