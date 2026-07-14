import networkx as nx
from typing import List
from langchain_community.graphs.graph_document import GraphDocument

from src.utils.logging import get_logger

logger = get_logger(__name__)

def build_graph(graph_documents: List[GraphDocument], embedder) -> nx.DiGraph:
    """Build a directed knowledge graph from langchain GraphDocument objects."""
    logger.info(f"Building knowledge graph from {len(graph_documents)} graph documents.")

    graph = nx.DiGraph()

    for gd in graph_documents:
        for rel in gd.relationships:
            graph.add_node(rel.source.id, type=rel.source.type)
            graph.add_node(rel.target.id, type=rel.target.type)
            graph.add_edge(rel.source.id, rel.target.id, relation=rel.type)


    node_labels = list(graph.nodes())
    vectors = embedder.embed(node_labels)
    for node, vector in zip(node_labels, vectors):
        graph.nodes[node]["embedding"] = vector


    logger.info(f"Graph built | nodes: {graph.number_of_nodes()} | edges: {graph.number_of_edges()}")
    return graph