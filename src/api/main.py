from contextlib import asynccontextmanager
from typing import Union, List, Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.utils.logging import get_logger

logger = get_logger("src.api")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

embedding_model = None
reranker_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global embedding_model
    global reranker_model

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info(f"Loading reranker model: {RERANKER_MODEL}...")
    reranker_model = CrossEncoder(RERANKER_MODEL)

    logger.info("Models loaded successfully.")

    yield

    embedding_model = None
    reranker_model = None


app = FastAPI(
    title="TriRAG Model Service",
    lifespan=lifespan
)


class EmbedRequest(BaseModel):
    text: str


class EmbedBatchRequest(BaseModel):
    texts: list[str]


class RerankRequest(BaseModel):
    question: str
    candidates: list[Union[dict, str]]
    top_k: int = 3


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_model_loaded": embedding_model is not None,
        "reranker_model_loaded": reranker_model is not None
    }


@app.post("/embed")
def embed(request: EmbedRequest):

    vector = embedding_model.encode(
        request.text,
        normalize_embeddings=True
    )

    return {
        "vector": vector.tolist()
    }


@app.post("/embed_batch")
def embed_batch(request: EmbedBatchRequest):

    vectors = embedding_model.encode(
        request.texts,
        normalize_embeddings=True
    )

    return {
        "vectors": vectors.tolist()
    }


@app.post("/rerank")
def rerank(request: RerankRequest):
    if not request.candidates:
        return {"results": []}

    pairs = [
        [request.question, candidate["text"] if isinstance(candidate, dict) else str(candidate)]
        for candidate in request.candidates
    ]

    scores = reranker_model.predict(pairs)

    ranked = []

    for candidate, score in zip(request.candidates, scores):
        if isinstance(candidate, dict):
            result = candidate.copy()
            result["rerank_score"] = float(score)
            ranked.append((result, float(score)))
        else:
            ranked.append((candidate, float(score)))

    ranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "results": [item for item, _ in ranked[:request.top_k]]
    }
