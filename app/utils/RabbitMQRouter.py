from typing import Optional
from app.types import EventHandler, QueueConfig

class RabbitMQRouter:
    def __init__(self, config: QueueConfig) -> None:
        self.config = config
        self.config.handlers = self.config.handlers or {}

    def route(self, name: str, handler: EventHandler) -> None:
        if not name:
            raise ValueError("Event name cannot be empty")
        if not handler or not callable(handler):
            raise ValueError("Handler must be a valid function")
        if name in self.config.handlers:
            print(f"Warning: Overwriting handler for {name}")
        self.config.handlers = {
            **self.config.handlers,
            name: handler
        }

    def remove_route(self, name: str) -> None:
        if name in self.config.handlers:
            self.config.handlers = {
                k: v for k, v in self.config.handlers.items() if k != name
            }

    def get_handler(self, name: str) -> Optional[EventHandler]:
        return self.config.handlers.get(name)
