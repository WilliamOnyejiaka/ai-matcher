from typing import Generator, List, Tuple
from app.config.db import db
from app.utils.reformat_to_bio import reformat_to_bio
from app.config.ai import model, index
from app.config.logger import logger
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo import UpdateOne


class AI:

    @classmethod
    def json_to_embeddings(cls, json_data: List[dict], batch_size: int = 100
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

    @classmethod
    def json_to_embedding(cls, json_data: dict) -> List:
        result = reformat_to_bio(json_data)
        embeddings = model.encode(result['bio'])
        return {"_id":  str(result['_id']), "embedding": embeddings.tolist()}
        # return [(result['_id'], embeddings, {"_id": str(result['_id'])})]

    async def update_embeddings(
        cls,
        json_data: List[dict],
        batch_size: int = 100,
    ) -> int:
        total_updated = 0
        collection = db["users"]

        for embeddings, ids in cls.json_to_embeddings(json_data, batch_size):
            try:
                operations = [
                    UpdateOne(
                        {"_id": ObjectId(_id)},               # filter
                        {"$set": {"embedding": vec}},         # update
                        upsert=False,
                    )
                    for _id, vec in zip(ids, embeddings)
                ]

                if not operations:
                    continue

                # Execute async bulk write
                result = await collection.bulk_write(
                    operations,
                    ordered=False,          # continue on individual errors
                )

                batch_count = result.modified_count
                total_updated += batch_count
                logger.info(
                    f"Updated {batch_count} embeddings (total so far: {total_updated})"
                )
            except Exception as e:
                logger.error(f"Failed to update : {e}")

        logger.info(
            f"Async embedding update complete: {total_updated} documents updated."
        )
        return total_updated

    @classmethod
    def search(cls, query_embedding_arr: list[float], embeddings_arr: list[dict], top_k: int = 5) -> None:
        query_embedding = np.array(query_embedding_arr, dtype=np.float32)
        embeddings = [np.array(doc["embedding"], dtype=np.float32)
                      for doc in embeddings_arr]

        # Cosine similarity between query and every candidate
        similarities = cosine_similarity([query_embedding], embeddings)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]

        print(
            f"\nTop {top_k} most similar results (out of {len(embeddings)} candidates):")
        result = []
        for idx, score in zip(top_indices, top_scores):
            match = embeddings_arr[idx]
            doc_id = match["_id"]
            del match["embedding"]
            result.append(match)
            print(
                f"  • Index {idx} | _id: {doc_id} | similarity = {score:.4f}")

        return result
