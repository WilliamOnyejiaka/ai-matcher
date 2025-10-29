import threading
from typing import Generator, List, Tuple
from sentence_transformers import SentenceTransformer
from app.config.logger import logger
from app.config.env import PINECONE_KEY


MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """Thread-safe lazy loading of the embedding model."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info(f"Loading SentenceTransformer: {MODEL_NAME}")
                # Force CPU – Render free tier has no GPU
                _model = SentenceTransformer(
                    MODEL_NAME,
                    device="cpu", 
                )
                logger.info("Model loaded successfully")
    return _model
