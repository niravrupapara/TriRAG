import os
import pickle
import networkx as nx
from src.utils.logging import get_logger

logger = get_logger(__name__)

class GraphStore:
    """Save and load the knowledge graph to disk."""

    def __init__(self, path: str):
        self.path = path
        self.graph_path = os.path.join(path, "graph.pkl")
        logger.debug("GraphStore initialized.")

    def save(self, graph: nx.DiGraph) -> None:
        """Save the knowledge graph to disk."""
        os.makedirs(self.path, exist_ok=True)

        with open(self.graph_path,'wb') as f:
            pickle.dump(graph, f)

        logger.info(f"Graph saved | nodes: {graph.number_of_nodes()} | edges: {graph.number_of_edges()}")

    def load(self) -> nx.DiGraph:
        """Load the knowledge graph from disk."""
        if not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"Graph not found at: {self.graph_path}")
        
        with open(self.graph_path, 'rb') as f:
            graph  = pickle.load(f)

        logger.info(f"Graph loaded | nodes: {graph.number_of_nodes()} | edges: {graph.number_of_edges()}")
        return graph