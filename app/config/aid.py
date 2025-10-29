from typing import Generator, List, Tuple
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from app.config.logger import logger
from app.config.env import PINECONE_KEY

pc = Pinecone(api_key=PINECONE_KEY)
index_name = "possible-matches"  # Fixed: Removed underscore for Pinecone compliance
model_name = "all-MiniLM-L6-v2"

# Create index if it doesn't exist (assuming 384 dimensions for all-MiniLM-L6-v2)
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 embedding size
        metric="cosine",  # Suitable for text embeddings
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    logger.info(f"Created new index: {index_name}")
else:
    # Optional: Verify existing index specs (e.g., dimension) to avoid mismatches
    index_desc = pc.describe_index(index_name)
    if index_desc.dimension != 384:
        raise ValueError(
            f"Existing index '{index_name}' has dimension {index_desc.dimension}, expected 384. Delete and recreate.")
    logger.info(f"Using existing index: {index_name}")

index = pc.Index(index_name)
model = SentenceTransformer(model_name)


# index=None
# model=None