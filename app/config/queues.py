from app.handlers.some import some
from app.handlers.user import user
from app.constants import QueueName

QUEUES = {
    QueueName.SOME_QUEUE: some.config,
    QueueName.USER_QUEUE: user.config

}
