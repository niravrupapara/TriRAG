import networkx as nx
from typing import List, Tuple
from src.utils.logging import get_logger

logger = get_logger(__name__)

def build_graph(triples: List[Tuple[str, str, str]]) -> nx.DiGraph:
    """Build a directed knowledge graph from a list of triples."""
    logger.info(f"Building knowledge graph from {len(triples)} triples.")

    graph = nx.DiGraph()

    for subject, predicate, obj in triples:
        graph.add_node(subject)
        graph.add_node(obj)
        graph.add_edge(subject, obj, relation=predicate)

    logger.info(f"Graph built | nodes: {graph.number_of_nodes()} | edges: {graph.number_of_edges()}")
    return graph