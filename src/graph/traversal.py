import numpy as np
import networkx as nx
from typing import List

from src.llm.llm_client import extract_entities
from src.utils.logging import get_logger

logger = get_logger(__name__)

def get_relevant_nodes(query: str, graph: nx.DiGraph, top_k: int, embedder, config: dict) -> List[str]:
    """Find graph nodes most similar to query entities via embedding similarity."""
    entities = extract_entities(query, config)
    search_terms = entities if entities else [query]

    node_labels = list(graph.nodes())
    node_vectores = np.array([graph.nodes[n]["embedding"] for n in node_labels])

    matched = []

    for term in search_terms:
        term_vector = embedder.embed_one(term)
        scores = np.dot(node_vectores, term_vector)
        top_indices = np.argsort(scores)[::-1][:top_k]

        for i in top_indices:
            if node_labels[i] not in matched:
                matched.append(node_labels[i])
    
    logger.debug(f"Query matched {len(matched)} nodes in graph | search terms: {search_terms} ")

    return matched[:top_k]


def traverse_subgraph(start_nodes: List[str], graph: nx.DiGraph, max_hops: int) -> List[str]:
    """Collect triples within max_hops of start nodes as text chunks."""
    logger.debug(f"Traversing graph from {len(start_nodes)} nodes | max_hops: {max_hops}")

    visited = set()
    chunks = []

    for start in start_nodes:
        if start not in graph:
            continue
        neighbors = nx.ego_graph(graph, start, radius=max_hops, undirected=True)
        for u, v, data in neighbors.edges(data=True):
            edge_key = (u, v)
            if edge_key not in visited:
                visited.add(edge_key)
                relation = data.get("relation", "related to")
                chunks.append(f"{u} {relation} {v}")

    logger.debug(f"Subgraph traversal complete | chunks collected: {len(chunks)}")
    return chunks

def query_graph(query: str, graph: nx.DiGraph, config: dict, embedder) -> List[dict]:
    """Find relevant subgraph for a query and return as list of chunks."""
    logger.info(f"GraphRAG querying for: {query[:60]}...")

    top_k = config["graph"]["top_k"]
    max_hops = config["graph"]["max_hops"]

    start_nodes = get_relevant_nodes(query, graph, top_k, embedder, config)

    if not start_nodes:
        logger.warning("No matching nodes found in graph for query.")
        return []

    chunks = traverse_subgraph(start_nodes, graph, max_hops)
    # results = [{"chunk": chunk, "score": 1.0} for chunk in chunks[:top_k]]

    if not chunks:
        logger.warning("No chunks collected from traversal")
        return []
    
    chunk_vectors = embedder.embed(chunks)
    query_vector = embedder.embed_one(query)
    scores = np.dot(chunk_vectors, query_vector)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [{"chunk": chunks[i], "score": float(scores[i])} for i in top_indices]

    logger.info(f"Graph query complete | results: {len(results)}")
    return results
