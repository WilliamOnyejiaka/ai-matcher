from typing import Any

from app.types import QueueConfig
from app.utils.RabbitMQRouter import RabbitMQRouter
from app.constants import QueueName, exchange
from app.services.AI import AI

user = RabbitMQRouter(QueueConfig(
    name=QueueName.USER_QUEUE.value,
    durable=True,
    routing_key_pattern="user_ai.*",
    exchange=exchange,
    handlers={}
))


async def upsert_embeddings(payload: Any, io: Any) -> None:
    successful = AI.upsert_vectordb([payload['payload']])
    print("✅ Data has been inserted") if successful else print(
        "🤷 Some upsert may have failed")


async def delete_embeddings(payload: Any, io: Any) -> None:
    successful = AI.delete_embeddings([payload['payload']])
    print("✅ Embeddings has been deleted successfully") if successful else print(
        "🛑 Failed to deleted embeddings")


user.route("user_ai.upsert_embeddings", upsert_embeddings)
user.route("user_ai.delete_embeddings", delete_embeddings)
