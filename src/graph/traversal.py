import numpy as np
import networkx as nx


def query_graph(query, graph, chunks, embedder, top_k=5, max_hops=2):
    if not chunks:
        return []

    query_vector = np.array(embedder.embed_query(query))
    chunk_vectors = np.array([chunk["embedding"] for chunk in chunks])

    scores = np.dot(chunk_vectors, query_vector)
    indices = np.argsort(scores)[::-1][:top_k]

    results = []
    seen = set()

    for index in indices:
        chunk_text = chunks[index]["text"]

        if chunk_text not in seen:
            seen.add(chunk_text)
            results.append(chunk_text)

        for node in graph.nodes:
            if node.lower() not in chunk_text.lower():
                continue

            subgraph = nx.ego_graph(
                graph,
                node,
                radius=max_hops,
                undirected=True
            )

            for _, _, data in subgraph.edges(data=True):
                chunk = data.get("chunk")

                if chunk and chunk not in seen:
                    seen.add(chunk)
                    results.append(chunk)

    return results[:top_k]