from typing import Generator, List, Tuple
from pymongo import MongoClient, GEOSPHERE
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer
from junk.bio import reformat_to_bio
from junk.data import user, data
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1. MongoDB Atlas connection
# ----------------------------------------------------------------------
ATLAS_URI = "mongodb+srv://wonder:william4000@cluster0.b4fmr.mongodb.net/blumdate_db?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(ATLAS_URI)
db = client["blumdate_db"]          # any DB name you like
collection = db["ai_test"]                 # any collection name

# ----------------------------------------------------------------------
# 2. Create a Vector Search index (once)
# ----------------------------------------------------------------------
INDEX_NAME = "vector_index"
DIMENSION = 384          # all-MiniLM-L6-v2


# def ensure_vector_index():
#     """Create a search index that supports knnBeta + complex filters."""
#     existing = [idx["name"] for idx in collection.list_search_indexes()]
#     if INDEX_NAME in existing:
#         logger.info(f"Search index '{INDEX_NAME}' already exists.")
#         return

#     index_model = SearchIndexModel(
#         definition={
#             "mappings": {
#                 "dynamic": False,  # Set True to index all fields automatically
#                 "fields": {
#                     "embedding": {
#                         "type": "knnVector",
#                         "dimensions": DIMENSION,
#                         "similarity": "cosine"
#                     },
#                     "_id": {
#                         "type": "string"  # Required for filtering on _id
#                     }
#                 }
#             }
#         },
#         name=INDEX_NAME,
#         type="search"  # Required for knnBeta
#     )

#     try:
#         result = collection.create_search_indexes([index_model])
#         logger.info(f"Created search index: {result}")
#     except Exception as e:
#         logger.error(f"Failed to create index: {e}")
#         raise



def ensure_vector_index():
    """Create a search index that supports knnBeta + complex filters."""
    existing = [idx["name"] for idx in collection.list_search_indexes()]
    if INDEX_NAME in existing:
        logger.info(f"Search index '{INDEX_NAME}' already exists.")
        return

    # index_model = SearchIndexModel(
    #     definition={
    #         "mappings": {
    #             "dynamic": True,                     # index every field automatically
    #             "fields": {
    #                 "embedding": {
    #                     "type": "vector",            # NEW
    #                     "numDimensions": DIMENSION,  # NEW
    #                     "similarity": "cosine"
    #                 }
    #             }
    #         }
    #     },
    #     name=INDEX_NAME,
    # )

    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": DIMENSION,
                    "similarity": "cosine",
                    "quantization": "scalar"
                }
            ]
        },
        name=INDEX_NAME,
        type="vectorSearch"
    )

    try:
        result = collection.create_search_indexes([search_index_model])

        collection.create_index([('location', GEOSPHERE)])
        logger.info(f"Created search index: {result}")
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        raise


def ensure_vector_index1():
    """Create a search index that supports knnBeta + complex filters."""
    existing = [idx["name"] for idx in collection.list_search_indexes()]
    if INDEX_NAME in existing:
        logger.info(f"Search index '{INDEX_NAME}' already exists.")
        return

    search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": DIMENSION,
                    "similarity": "cosine",
                    "quantization": "scalar"
                },
                # "location": {  # Geospatial field (GeoJSON object)
                #     "type": "object",  # Or use "dynamic": true if schema varies
                #     "fields": {  # Optional: Index subfields if needed (e.g., for autocomplete)
                #             "type": {"type": "string"},
                #             # For [lng, lat]
                #             "coordinates": {"type": "double", "numDims": 2}
                #     }
                # }
            }
        },
        name=INDEX_NAME,
        type="vectorSearch",
    )

    try:
        result = collection.create_search_indexes([search_index_model])

        collection.create_index([('location', GEOSPHERE)])
        logger.info(f"Created search index: {result}")
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        raise


ensure_vector_index()

# ----------------------------------------------------------------------
# 3. Embedding model
# ----------------------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------------------------------------------------
# 4. Helper: batch embeddings (unchanged)
# ----------------------------------------------------------------------


def json_to_embeddings(json_data: List[dict], batch_size: int = 100
                       ) -> Generator[Tuple[List, List], None, None]:
    """Yield (embeddings, ids) for each batch."""
    for i in range(0, len(json_data), batch_size):
        batch = json_data[i:i + batch_size]
        bios = [reformat_to_bio(item)['bio'] for item in batch]
        try:
            batch_emb = model.encode(bios, batch_size=batch_size)
            ids = [item['_id'] for item in batch]
            yield batch_emb.tolist(), ids
        except Exception as e:
            logger.error(f"Embedding batch {i//batch_size} failed: {e}")
            continue

# ----------------------------------------------------------------------
# 5. Insert / upsert vectors into MongoDB
# ----------------------------------------------------------------------


def insert_to_vectordb(json_data: List[dict], batch_size: int = 100):
    """Upsert documents that contain the vector in field `embedding`."""
    for embeddings, ids in json_to_embeddings(json_data, batch_size):
        docs = []
        for _id, vec in zip(ids, embeddings):
            # Keep everything you already have in the original dict + the vector
            orig = next(d for d in json_data if d["_id"] == _id)
            doc = orig.copy()
            doc["embedding"] = vec
            docs.append(doc)

        # Upsert: match on _id, replace whole doc
        for doc in docs:
            collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)

        logger.info(f"Upserted batch of {len(docs)} vectors")

# ----------------------------------------------------------------------
# 6. Vector search (cosine, top-k, optional metadata filter)
# ----------------------------------------------------------------------


# def search_vectordb(
#     bio: str,
#     n: int = 1,
#     exclude_ids: List[str] | None = None,
# ) -> List[str]:
#     if not bio.strip():
#         logger.warning("Empty query bio → returning []")
#         return []

#     query_vec = model.encode([bio])[0].tolist()

#     knn: dict = {
#         "vector": query_vec,
#         "path": "embedding",
#         "k": n * 10,                     # broader recall
#     }

#     # if exclude_ids:
#     #     knn["filter"] = {
#     #         "compound": {
#     #             "mustNot": [{"in": {"path": "_id", "value": exclude_ids}}]
#     #         }
#     #     }

#     # pipeline = [
#     #     {"$search": {"index": INDEX_NAME, "knnBeta": knn}},
#     #     {"$limit": n},
#     #     {"$project": {"_id": 1, "score": {"$meta": "searchScore"}}},

#     pipeline = [
#         {"$search": {"index": "vector_index", "knnBeta": {
#             "vector": query_vec, "path": "embedding", "k": 500}}},
#         {"$limit": 10},
#         {"$project": {"_id": 1, "score": {"$meta": "searchScore"}}}
#     ]
#     # ]

#     try:
#         return [str(r["_id"]) for r in collection.aggregate(pipeline)]
#     except Exception as exc:
#         logger.error(f"knnBeta search failed: {exc}")
#         return []


def search_vectordb(
    bio: str,
    n: int = 1,
    exclude_ids: List[str] | None = None,
) -> List[str]:
    if not bio.strip():
        logger.warning("Empty query bio → returning []")
        return []
    user_id = "user0"
    # user = collection.find_one({'_id': user_id})
    # coordinates = user['location']['coordinates']
    # gender = user['gender']

    query_vec = model.encode([bio])[0].tolist()

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vec,
                "numCandidates": 200,
                "limit": 10
            }
        }, {
            '$project': {
                '_id': 1,
                # 'plot': 1,
                # 'title': 1,
                'score': {
                    '$meta': 'vectorSearchScore'
                }
            }
        }
    ]
    # ]

    try:
        return [str(r["_id"]) for r in collection.aggregate(pipeline)]
    except Exception as exc:
        logger.error(f"knnBeta search failed: {exc}")
        return []


def search_vectordb1(
    bio: str,
    n: int = 1,
    exclude_ids: List[str] | None = None,
) -> List[str]:
    if not bio.strip():
        logger.warning("Empty query bio → returning []")
        return []
    user_id = "user0"
    # user = collection.find_one({'_id': user_id})
    # coordinates = user['location']['coordinates']
    # gender = user['gender']

    query_vec = model.encode([bio])[0].tolist()
    user_location = {'type': 'Point',
                     'coordinates': [-73.985, 40.748]}  # LA

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vec,
                "numCandidates": 200,
                "limit": 10,
                'filter': {  # Geospatial pre-filter
                    'location': {
                        '$nearSphere': user_location,
                        '$maxDistance': 50000  # 50km in meters
                    }
                }
            }
        }, {
            '$project': {
                '_id': 1,
                # 'plot': 1,
                # 'title': 1,
                'score': {
                    '$meta': 'vectorSearchScore'
                }
            }
        }
    ]
    # ]

    try:
        return [str(r["_id"]) for r in collection.aggregate(pipeline)]
    except Exception as exc:
        logger.error(f"knnBeta search failed: {exc}")
        return []
    

# def search(query_embedding_arr, embeddings_arr):
#     query_embedding = np.array(query_embedding_arr)
#     embeddings = []
#     for embedding in embeddings_arr:
#         embeddings.append(np.array(embedding))

#     # Your vector search code
#     similarities = cosine_similarity([query_embedding], embeddings)[0]
#     top_k = 5
#     top_indices = np.argsort(similarities)[-top_k:][::-1]
#     top_scores = similarities[top_indices]

#     print(f"Top {top_k} most similar results:")
#     for idx, score in zip(top_indices, top_scores):
#         print(f"Index {idx}: similarity={score:.4f}")

def search1(query_embedding_arr, embeddings_arr):
    query_embedding = np.array(query_embedding_arr)
    embeddings = []
    for embedding in embeddings_arr:
        embeddings.append(np.array(embedding['embedding']))

    # Your vector search code
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_k = 5
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_scores = similarities[top_indices]

    print(f"Top {top_k} most similar results:")
    for idx, score in zip(top_indices, top_scores):
        print(f"Index {idx}: similarity={score:.4f}")


def search(query_embedding_arr: list[float], embeddings_arr: list[dict]) -> None:
    """
    `embeddings_arr` is a list of dicts returned by location_search(),
    each containing an `_id` and an `embedding` list.
    """
    query_embedding = np.array(query_embedding_arr, dtype=np.float32)
    embeddings = [np.array(doc["embedding"], dtype=np.float32)
                  for doc in embeddings_arr]

    # Cosine similarity between query and every candidate
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    top_k = 5
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_scores = similarities[top_indices]

    print(
        f"\nTop {top_k} most similar results (out of {len(embeddings)} candidates):")
    for idx, score in zip(top_indices, top_scores):
        doc_id = embeddings_arr[idx]["_id"]
        print(f"  • Index {idx} | _id: {doc_id} | similarity = {score:.4f}")

def location_search():
    # user = .user_collection.find_one({'_id': ObjectId(user_id)})
    # coordinates = user.get("location", {}).get("coordinates")
    # gender = user.get("gender", {})

    pipeline = [
        {
            "$geoNear": {
                "near": {"type": "Point", "coordinates": [-73.985, 40.748]},
                # Include distance in output (in meters)
                "distanceField": "distance",
                "maxDistance": 10 * 1000,  # Convert km to meters
                "spherical": True,  # Required for 2dsphere index
                "query": {
                    "status": "active",
                    "_id": {"$ne": "user0"},
                    # Add gender filter if provided
                    **({"genderInterest": "male"})
                }
            }
        },
        {"$project": {"_id": 1,"embedding": 1}},
        {"$limit": 20}
    ]

    cursor = collection.aggregate(pipeline).to_list()
    # return [dict(user, _id=str(user['_id'])) for user in cursor.to_list()]
    return cursor


# ----------------------------------------------------------------------
# 7. Demo (identical to your original Pinecone demo)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    n_matches = 100
    bio = reformat_to_bio(user)['bio']

    # # insert_to_vectordb(data)

    # results = search_vectordb(
    #     bio,
    #     # data,
    #     n=n_matches,
    #     # exclude_ids=["user8", "user7"]
    # )
    # sample_doc = collection.find_one({"embedding": {"$exists": True}}, {
    #                                  "_id": 1, "embedding": 1})  # First doc with embedding
    # print(len(sample_doc['embedding']))

    location_query = location_search()
    query_vec = model.encode([bio])[0].tolist()

    search(query_vec, location_query)

    # print("Top matched _id's:", location)
