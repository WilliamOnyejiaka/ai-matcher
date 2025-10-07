from typing import Generator, List, Tuple
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from bio import reformat_to_bio
from data import user, data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
pinecone_key = "pcsk_52x6tX_CRBvDccPtuwFj4MJAHRNfffYwSQNjgDaNEtF4GUv2bQG5hdbdtuQM4hdNpKNThm"

pc = Pinecone(api_key=pinecone_key)
index_name = "possible-matches"  # Fixed: Removed underscore for Pinecone compliance

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
model = SentenceTransformer("all-MiniLM-L6-v2")


def json_to_embeddings(json_data: List[dict], batch_size: int = 100) -> Generator[Tuple[List, List], None, None]:
    """Generate embeddings in batches, yielding incrementally."""
    for i in range(0, len(json_data), batch_size):
        batch = json_data[i:i + batch_size]
        bios = [reformat_to_bio(item)['bio'] for item in batch]
        try:
            batch_embeddings = model.encode(bios, batch_size=batch_size)
            ids = [item['_id'] for item in batch]
            yield batch_embeddings.tolist(), ids
        except Exception as e:
            logger.error(
                f"Error generating embeddings for batch {i//batch_size}: {str(e)}")
            continue


def insert_to_vectordb(json_data: List[dict], batch_size: int = 100):
    # Preprocess and insert new data
    for embeddings, ids in json_to_embeddings(json_data, batch_size):
        try:
            vectors = [(id, emb) for id, emb in zip(ids, embeddings)]
            index.upsert(vectors=vectors)
            logger.info(f"Inserted batch of {len(vectors)} vectors.")
        except Exception as e:
            logger.error(f"Failed to insert batch to Pinecone: {str(e)}")


def search_vectordb(bio: str, json_data: List[dict], n: int = 1) -> List[str]:
    """Search Pinecone for top-n matches, insert new data if needed."""
    insert_to_vectordb(json_data)  # Insert any new data
    try:
        query_embedding = model.encode([bio])[0].tolist()
        results = index.query(vector=query_embedding,
                              top_k=n, include_values=False)
        matched_ids = [result.id for result in results.matches]

        if matched_ids:
            index.delete(ids=matched_ids)

        return matched_ids
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return []


if __name__ == "__main__":
    n = 100
    bio = reformat_to_bio(user)['bio']
    results = search_vectordb(bio, data, n)
    print(results)
