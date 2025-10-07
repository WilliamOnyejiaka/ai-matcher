import json
from typing import Dict, Optional, Any
import aio_pika
from aio_pika import ExchangeType, Message
from app.config.env import RABBITMQ_URL
from app.config.logger import logger
from app.config.queues import QUEUES
from app.constants import QueueName


class RabbitMQ:
    _connection: Optional[aio_pika.RobustConnection] = None
    _channels: Dict[QueueName, aio_pika.RobustChannel] = {}
    PREFETCH_COUNT = 1  # Process one message at a time per consumer
    RABBITMQ_URL = RABBITMQ_URL

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
                f"Message sent to {QUEUES[queue_name].exchange} with routing key {event_type}")
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
                            f"📥 Received on {queue_name}, event type - {event_type}")
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
