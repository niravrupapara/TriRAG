from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks for a query."""
        pass
