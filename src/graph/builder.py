
import networkx as nx
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_graph(graph_documents):
    logger.info(f"Building knowledge graph from {len(graph_documents)} graph documents")
    graph = nx.DiGraph()

    for document in graph_documents:
        chunk = document.source.page_content if document.source else ""

        for rel in document.relationships:
            if not rel.source.id or not rel.target.id:
                continue

            graph.add_node(rel.source.id)
            graph.add_node(rel.target.id)

            graph.add_edge(
                rel.source.id,
                rel.target.id,
                relation=rel.type,
                chunk=chunk
            )

    logger.info(f"Graph constructed: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph