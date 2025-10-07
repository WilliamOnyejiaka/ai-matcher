import asyncio
from enum import Enum
import uvicorn
from app.config.app import create_app
from app.config.env import PORT
from app.constants import QueueName
from app.services.RabbitMQ import RabbitMQ

app = create_app()

async def connect_to_rabbitMQ():
    await RabbitMQ.connect()
    for queue in QueueName:
        await RabbitMQ.start_consumer(queue)


async def main():
    tasks = [
        asyncio.create_task(connect_to_rabbitMQ())
    ]

    #* Start Uvicorn server in the same event loop
    config = uvicorn.Config(app, host="0.0.0.0", port=int(PORT))
    server = uvicorn.Server(config)
    await server.serve()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await RabbitMQ.close()

if __name__ == "__main__":
    asyncio.run(main())
