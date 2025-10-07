from typing import Any

from app.types import QueueConfig
from app.utils.RabbitMQRouter import RabbitMQRouter
from app.constants import QueueName


async def some_handler(payload: Any, io: Any):
    print(f"Handling payload: {payload}")


some = RabbitMQRouter(QueueConfig(
    name=QueueName.SOME_QUEUE.value,
    durable=True,
    routing_key_pattern="some.*",
    exchange="some_exchange",
    handlers={}
))

some.route("some.event", some_handler)
