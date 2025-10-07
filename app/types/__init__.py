from typing import Any, Callable, Dict

EventHandler = Callable[[Any, Any], None]

class QueueConfig:
    def __init__(
        self,
        name: str,
        durable: bool,
        routing_key_pattern: str,
        exchange: str,
        handlers: Dict[str, EventHandler]
    ):
        self.name = name
        self.durable = durable
        self.routing_key_pattern = routing_key_pattern
        self.exchange = exchange
        self.handlers = handlers
