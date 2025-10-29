from typing import Any
from bson.objectid import ObjectId
from app.config.logger import logger
from app.config.db import db
from app.types import QueueConfig
from app.utils.RabbitMQRouter import RabbitMQRouter
from app.constants import QueueName, exchange
from app.services.AI import AI
from bson.json_util import dumps

user = RabbitMQRouter(QueueConfig(
    name=QueueName.USER_QUEUE.value,
    durable=True,
    routing_key_pattern="user_ai.*",
    exchange=exchange,
    handlers={}
))


async def upsert_embeddings(payload: Any, io: Any) -> None:
    successful = AI.upsert_vectordb_bulk([payload['payload']])
    print("✅ Data has been inserted") if successful else print(
        "🤷 Some upsert may have failed")


async def delete_embeddings(payload: Any, io: Any) -> None:
    successful = AI.delete_embeddings([payload['payload']])
    print("✅ Embeddings has been deleted successfully") if successful else print(
        "🛑 Failed to deleted embeddings")


async def embed(message: Any, io: Any) -> None:
    user = message['payload']

    print(user)

    embedding_result = AI.json_to_embedding(user)
    collection = db["users"]
    user_id = embedding_result['_id']

    try:
        result = await collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'embedding': embedding_result['embedding']}}
        )

        if result.matched_count > 0:
            print(f"👍 User embeddings has been updated")
        else:
            print(f"🤷 No document found with the given {user_id}")
    except Exception as e:
        logger.error(f"🛑 Failed to update embeddings : {e}")


user.route("user_ai.embed", embed)
user.route("user_ai.upsert_embeddings", upsert_embeddings)
user.route("user_ai.delete_embeddings", delete_embeddings)
