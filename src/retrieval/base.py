from abc import ABC, abstractmethod
from typing import List

class BaseRetriever(ABC):
    """Abstract base class that all retrievers must implement."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[dict]:
        """Retrieve relevant documents based on the query."""
        pass