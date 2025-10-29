from typing import List

import motor
from app.config.db import db
import asyncio
from bson import ObjectId
from typing import Generator, List, Tuple
from pymongo import MongoClient, GEOSPHERE
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo import UpdateOne
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer
from junk.bio import reformat_to_bio
from junk.data import user, data
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.services.Recommendation import Recommendation 

collection = db["users"]
# model = SentenceTransformer("all-MiniLM-L6-v2")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
recommend = Recommendation()


# def json_to_embeddings(json_data: List[dict], batch_size: int = 100
#                        ) -> Generator[Tuple[List, List], None, None]:
#     """Yield (embeddings, ids) for each batch."""
#     for i in range(0, len(json_data), batch_size):
#         batch = json_data[i:i + batch_size]
#         bios = [reformat_to_bio(item)['bio'] for item in batch]
#         try:
#             batch_emb = model.encode(bios, batch_size=batch_size)
#             ids = [item['_id'] for item in batch]
#             yield batch_emb.tolist(), ids
#         except Exception as e:
#             logger.error(f"Embedding batch {i//batch_size} failed: {e}")
#             continue

# async def update_embeddings1(json_data: List[dict], batch_size: int = 100):
#     """
#     Async: Update ONLY the 'embedding' field using Motor + $set + bulk_write.
#     """
#     total_updated = 0

#     for embeddings, ids in json_to_embeddings(json_data, batch_size):
#         # Build bulk operations
#         operations = [
#             UpdateOne(
#                 {"_id": ObjectId(_id)},               # filter
#                 {"$set": {"embedding": vec}},         # update
#                 upsert=False,
#             )
#             for _id, vec in zip(ids, embeddings)
#         ]

#         if not operations:
#             continue

#         # Motor bulk_write equivalent
#         result = await collection.bulk_write(operations
#                                              )

#         batch_count = result.modified_count
#         total_updated += batch_count
#         logger.info(
#             f"Updated {batch_count} embeddings (total: {total_updated})")

#     logger.info(
#         f"Async embedding update complete: {total_updated} documents updated.")


# async def get_users():
#     cursor = collection.find({})
#     result = [dict(user, _id=str(user['_id'])) for user in await cursor.to_list()]
#     return result


# async def update_embeddings(
#     json_data: List[dict],
#     batch_size: int = 100,
# ) -> int:
#     total_updated = 0

#     for embeddings, ids in json_to_embeddings(json_data, batch_size):
#         try:
#             operations = [
#                 UpdateOne(
#                     {"_id": ObjectId(_id)},               # filter
#                     {"$set": {"embedding": vec}},         # update
#                     upsert=False,
#                 )
#                 for _id, vec in zip(ids, embeddings)
#             ]

#             if not operations:
#                 continue

#             # Execute async bulk write
#             result = await collection.bulk_write(
#                 operations,
#                 ordered=False,          # continue on individual errors
#             )

#             batch_count = result.modified_count
#             total_updated += batch_count
#             logger.info(
#                 f"Updated {batch_count} embeddings (total so far: {total_updated})"
#             )
#         except Exception as e:
#             logger.error(f"Failed to update : {e}")


#     logger.info(
#         f"Async embedding update complete: {total_updated} documents updated."
#     )
#     return total_updated


def search(query_embedding_arr: list[float], embeddings_arr: list[dict]) -> None:
    query_embedding = np.array(query_embedding_arr, dtype=np.float32)
    embeddings = [np.array(doc["embedding"], dtype=np.float32)
                  for doc in embeddings_arr]

    # Cosine similarity between query and every candidate
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    top_k = 10
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_scores = similarities[top_indices]

    print(
        f"\nTop {top_k} most similar results (out of {len(embeddings)} candidates):")
    for idx, score in zip(top_indices, top_scores):
        doc_id = embeddings_arr[idx]["_id"]
        print(f"  • Index {idx} | _id: {doc_id} | similarity = {score:.4f}")

async def test():
    # result = await recommend.possible_matches("68cdc013137f27f7eca9cd8f",20)
    # matches = result['matches']
    # query_embedding_arr = result['user']['embedding']
    # # embeddings_arr = []
    # # for match in matches:
    # #     embeddings_arr.append(match['embedding'])
    
    # search_result = search(query_embedding_arr, matches)
    # print(search_result)
    # for embeddings, ids, bios in json_to_embeddings(users, 100):
    #     for _id, vec, bio in zip(ids, embeddings, bios):
    #         print(f'{_id} bio - {bio}')

    result = await recommend.recommend("68cdc013137f27f7eca9cd8f")
    print(result)

# Run the async function
if __name__ == "__main__":

    asyncio.run(test())
