try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop(
        'pysqlite3')  # Override default sqlite3
    import sqlite3
except ImportError:
    print("Error: pysqlite3-binary not installed. Run: pip install pysqlite3-binary")
    exit(1)

from pinecone import Pinecone, ServerlessSpec
from typing import Generator, List, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from bio import reformat_to_bio
from data import user, data
import logging
import pinecone

pinecone_key = "pcsk_52x6tX_CRBvDccPtuwFj4MJAHRNfffYwSQNjgDaNEtF4GUv2bQG5hdbdtuQM4hdNpKNThm"

pc = Pinecone(
    api_key=pinecone_key
)

match_collection = "possible_matches_collection"
index_name = "possible-matches"


# Now do stuff
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 embedding size
        metric="cosine",  # Suitable for text embeddings
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


index = pc.Index(index_name)

# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

client = chromadb.PersistentClient("./chroma_db")

model = SentenceTransformer("all-MiniLM-L6-v2")


def init_vectordb(collection_name: str):
    try:
        return client.get_or_create_collection(collection_name)
    except Exception as e:
        print(f"Failed to initialize ChromaDB collection: {str(e)}")
        raise


def json_to_embeddings(json_data: List[dict], batch_size: int = 100) -> Generator[Tuple[List, List], None, None]:
    """Generate embeddings in batches, yielding incrementally."""
    for i in range(0, len(json_data), batch_size):
        batch = json_data[i:i + batch_size]
        bios = [item['bio'] for item in batch]
        try:
            batch_embeddings = model.encode(bios, batch_size=batch_size)
            ids = [item['_id'] for item in batch]
            yield batch_embeddings.tolist(), ids
        except Exception as e:
            print(
                f"Error generating embeddings for batch {i//batch_size}: {str(e)}")
            continue


def insert_to_vectordb(json_data: List[dict], batch_size: int = 100):
    for i in range(0, len(json_data), batch_size):
        batch = json_data[i:i + batch_size]
        possible_matches_json = []
        for item in batch:
            reformatted = reformat_to_bio(item)
            possible_matches_json.append(reformatted)
        if possible_matches_json:
            for embeddings, ids in json_to_embeddings([reformat_to_bio(item) for item in json_data]):
                try:
                    vectors = [(id, emb) for id, emb in zip(ids, embeddings)]
                    index.upsert(vectors=vectors)
                except Exception as e:
                    print(f"Failed to insert batch to ChromaDB: {str(e)}")
        else:
            print(
                f"No valid data in batch {i//batch_size} to insert into vector DB.")


# def insert_to_vectordb(json_data: List[dict], batch_size: int = 100):
#     collection = init_vectordb(match_collection)
#     for i in range(0, len(json_data), batch_size):
#         batch = json_data[i:i + batch_size]
#         possible_matches_json = []
#         for item in batch:
#             reformatted = reformat_to_bio(item)
#             possible_matches_json.append(reformatted)
#         if possible_matches_json:
#             for embeddings, ids in json_to_embeddings([reformat_to_bio(item) for item in json_data]):
#                 try:
#                     collection.upsert(embeddings=embeddings, ids=ids)
#                 except Exception as e:
#                     print(f"Failed to insert batch to ChromaDB: {str(e)}")
#         else:
#             print(
#                 f"No valid data in batch {i//batch_size} to insert into vector DB.")

# def search_vectordb(bio: str, json_data, n=1):
#     insert_to_vectordb(json_data)

#     collection = init_vectordb(match_collection)
#     results = collection.query(query_texts=[bio], n_results=n)

#     matched_ids = results['ids'][0]

#     if matched_ids:
#         collection.delete(ids=matched_ids)
#     else:
#         print("No matches found; nothing deleted.")
#     return matched_ids

# def search_vectordb(bio: str, json_data, n=1):
#     insert_to_vectordb(json_data)  # Only run if new data needs indexing
#     query_embedding = model.encode([bio])[0].tolist()
#     try:
#         results = index.query(vector=query_embedding,
#                               top_k=n, include_values=False)
#         matched_ids = [result.id for result in results.matches]
#         return matched_ids
#     except Exception as e:
#         print(f"Query failed: {str(e)}")
#         return []

def search_vectordb(bio: str, json_data, n=1):
    insert_to_vectordb(json_data)  # Only run if new data needs indexing
    try:
        query_embedding = model.encode([bio])[0].tolist()
        results = index.query(vector=query_embedding,
                              top_k=n, include_values=False)
        matched_ids = [result.id for result in results.matches]
        return matched_ids
    except Exception as e:
        print(f"Query failed: {str(e)}")
        return []



if __name__ == "__main__":
    n = 100
    bio = reformat_to_bio(user)['bio']
    results = search_vectordb(bio, data, n)
    print(results)
