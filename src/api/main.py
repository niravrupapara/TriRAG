from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.engine import load_config
from src.embeddings.embedder import Embedder

from src.rerankers.cross_encoder import CrossEncoderReranker

from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model and reranker once when the server starts."""
    config = load_config("config.yaml")
    logger.info("Loading embedding model...")
    embedder = Embedder(config["embeddings"]["model"])

    logger.info("Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker(config["reranker"]["model"])

    app.state.embedder = embedder
    app.state.reranker = reranker
    app.state.config = config
    logger.info("Embedder + reranker ready, FastAPI startup complete.")

    yield


app = FastAPI(title = "TriRAG API", lifespan=lifespan)

@app.get("/")
def root():
    """Root endpoint to check if the API is running."""
    return {"message": "TriRAG API is running."}




class EmbedRequest(BaseModel):
    text : str
class EmbedResponse(BaseModel):
    text : str
    dimension : int
    preview : list

class EmbedBatchRequest(BaseModel):
    texts : list

class EmbedBatchResponse(BaseModel):
    vectors : list

class RerankerRequest(BaseModel):
    question : str
    candidates : list
    top_k : int

class RerankerResponse(BaseModel):
    results : list


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/embed", response_model=EmbedResponse)
def embed_text(request: EmbedRequest):
    """Generate embedding for the provided text."""
    vectore = app.state.embedder.embed_one(request.text)
    return EmbedResponse(
        text=request.text,
        dimension=len(vectore),
        preview=vectore[:5].tolist()  # Return first 5 dimensions as a preview
    )

@app.post("/embed_batch", response_model=EmbedBatchResponse)
def embed_batch(request: EmbedBatchRequest):
    """Embed multiple texts at once using the already-loaded embedding model."""
    vectors = app.state.embedder.embed(request.texts)
    return EmbedBatchResponse(vectors=vectors.tolist())

@app.post("/rerank", response_model=RerankerResponse)
def rerank_chunks(request: RerankerRequest):
    """Rerank candidate chunks for a question using the loaded cross-encoder reranker."""
    results = app.state.reranker.rerank(
        request.question,
        request.candidates,
        request.top_k
    )
    return RerankerResponse(results=results)
