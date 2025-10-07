import asyncio
import json
import logging
from typing import Dict, Optional, Any, Callable, TypeVar
from enum import Enum
import aio_pika
from aio_pika import ExchangeType, Message
# from config import env, logger  # Assuming environment and logger setup

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Equivalent of QueueConfig interface
# Equivalent of QueueConfig interface
T = TypeVar('T')


class QueueConfig:
    def __init__(
        self,
        name: str,
        durable: bool,
        routing_key_pattern: str,
        exchange: str,
        handlers: Dict[str, Callable[[Any, Any], None]]
    ):
        self.name = name
        self.durable = durable
        self.routing_key_pattern = routing_key_pattern
        self.exchange = exchange
        self.handlers = handlers

# Assuming QueueName is an Enum


class QueueName(Enum):
    SOME_QUEUE = "some_queue"  # Adjust as per your needs

# Assuming QUEUES is defined in config.queues
# Example structure:
# QUEUES = {
#     QueueName.SOME_QUEUE: QueueConfig(
#         name="some_queue",
#         durable=True,
#         routing_key_pattern="some.*",
#         exchange="some_exchange",
#         handlers={"some.event": some_handler_function}
#     )
# }


class RabbitMQ:
    _connection: Optional[aio_pika.RobustConnection] = None
    _channels: Dict[QueueName, aio_pika.RobustChannel] = {}
    PREFETCH_COUNT = 1  # Process one message at a time per consumer
    RABBITMQ_URL = "amqp://localhost:5672"  # Environment variable for RabbitMQ URL

    @classmethod
    async def connect(cls):
        if not cls._connection:
            try:
                cls._connection = await aio_pika.connect_robust(
                    cls.RABBITMQ_URL,
                    reconnect_interval=5  # Reconnect after 5 seconds
                )
                logger.info("Connected to RabbitMQ")
            except Exception as error:
                logger.error(f"Failed to connect to RabbitMQ: {error}")

    @classmethod
    async def get_channel(cls, queue_name: QueueName) -> aio_pika.RobustChannel:
        if not cls._connection:
            await cls.connect()

        if queue_name not in cls._channels:
            channel = await cls._connection.channel()
            cls._channels[queue_name] = channel

            # Assert the main topic exchange
            exchange = await channel.declare_exchange(
                QUEUES[queue_name].exchange,
                ExchangeType.TOPIC,
                durable=True
            )

            # Assert the dead-letter exchange
            dlx_name = f"{QUEUES[queue_name].exchange}_dlx"
            dlx = await channel.declare_exchange(dlx_name, ExchangeType.DIRECT, durable=True)

            # Assert the dead-letter queue
            dlq_name = f"{QUEUES[queue_name].name}.dlq"
            dlq = await channel.declare_queue(dlq_name, durable=True)

            # Bind the dead-letter queue to the dead-letter exchange
            await dlq.bind(dlx, routing_key=dlq_name)

            # Assert the main queue with dead-letter configuration
            queue = await channel.declare_queue(
                QUEUES[queue_name].name,
                durable=QUEUES[queue_name].durable,
                arguments={
                    "x-dead-letter-exchange": dlx_name,
                    "x-dead-letter-routing-key": dlq_name
                }
            )

            # Bind the main queue to the main exchange
            await queue.bind(
                exchange,
                routing_key=QUEUES[queue_name].routing_key_pattern
            )

            # Set prefetch limit
            await channel.set_qos(prefetch_count=cls.PREFETCH_COUNT)

            logger.info(
                f"Channel created for queue: {queue_name}, "
                f"bound to {QUEUES[queue_name].exchange} with pattern {QUEUES[queue_name].routing_key_pattern}, "
                f"prefetch: {cls.PREFETCH_COUNT}, DLX: {dlx_name}, DLQ: {dlq_name}"
            )

        return cls._channels[queue_name]

    @classmethod
    async def publish_to_exchange(cls, queue_name: QueueName, event_type: str, message: Any) -> bool:
        try:
            if queue_name not in QUEUES:
                raise ValueError(f"Invalid queue: {queue_name}")

            channel = await cls.get_channel(queue_name)
            exchange = await channel.get_exchange(QUEUES[queue_name].exchange)

            await exchange.publish(
                Message(
                    body=json.dumps(message).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=event_type
            )
            logger.info(
                f"Message sent to {QUEUES[queue_name].exchange} with routing key {event_type}: {message}")
            return True
        except Exception as error:
            logger.error(f"Failed to publish: {error}")
            return False

    @classmethod
    async def start_consumer(cls, queue_name: QueueName, io: Any = None):
        try:
            channel = await cls.get_channel(queue_name)

            async def on_message(message: aio_pika.IncomingMessage):
                try:
                    # Use process context manager to handle ack/reject automatically
                    async with message.process(requeue=False):
                        payload = json.loads(message.body.decode())
                        event_type = payload.get("eventType")
                        data = payload.get("payload")

                        if not event_type or event_type not in QUEUES[queue_name].handlers:
                            logger.error(
                                f"Unknown event_type: {event_type} in {queue_name}")
                            raise ValueError(
                                f"Unknown event_type: {event_type}")

                        logger.info(
                            f"Received on {queue_name} [{event_type}]: {data}")
                        await QUEUES[queue_name].handlers[event_type](payload, io)
                except Exception as err:
                    logger.error(
                        f"Error processing message on {queue_name}: {err}")
                    # No need to reject manually; message.process handles it on exception

            queue = await channel.get_queue(QUEUES[queue_name].name)
            await queue.consume(on_message)
            logger.info(f"Consumer started for {queue_name}")
        except Exception as err:
            logger.error(f"Consumer error for {queue_name}: {err}")

    @classmethod
    async def close(cls):
        try:
            for queue_name in list(cls._channels.keys()):
                await cls._channels[queue_name].close()
                del cls._channels[queue_name]
            if cls._connection:
                await cls._connection.close()
                cls._connection = None
            logger.info("RabbitMQ connection closed")
        except Exception as error:
            logger.error(f"Error closing RabbitMQ connection: {error}")

# Example usage


async def main():
    # Example handler function
    async def some_handler(payload: Any, io: Any):
        print(f"Handling payload: {payload}")

    # Example QUEUES configuration
    global QUEUES
    QUEUES = {
        QueueName.SOME_QUEUE: QueueConfig(
            name="some_queue",
            durable=True,
            routing_key_pattern="some.*",
            exchange="some_exchange",
            handlers={"some.event": some_handler}
        )
    }

    await RabbitMQ.connect()
    await RabbitMQ.start_consumer(QueueName.SOME_QUEUE)
    await RabbitMQ.publish_to_exchange(
        QueueName.SOME_QUEUE,
        "some.event",
        {"eventType": "some.event", "payload": {"data": "example"}}
    )
    # Keep the event loop running for testing
    try:
        # Keep running for 1 hour or adjust as needed
        await asyncio.sleep(3600)
    finally:
        await RabbitMQ.close()

if __name__ == "__main__":
    asyncio.run(main())
