from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from main import load_config
from src.embeddings.embedder import Embedder
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model once when the server starts."""
    config = load_config("config.yaml")
    logger.info("Loading embedding model...")
    embedder = Embedder(config["embeddings"]["model"])

    app.state.embedder = embedder
    app.state.config = config
    logger.info("Embedder ready, FastAPI startup complete.")

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
