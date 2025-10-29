
from enum import Enum

exchange="ai_exchange"

class QueueName(Enum):
    SOME_QUEUE = "some_queue"
    USER_QUEUE="user_ai_queue"

class ModelName(Enum):
    USER="users"