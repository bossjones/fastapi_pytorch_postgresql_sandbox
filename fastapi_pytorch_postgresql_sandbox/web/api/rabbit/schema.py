"""web.api.schema"""
# sourcery skip: avoid-global-variables
# pylint: disable=no-name-in-module

from typing import Optional

from pydantic import BaseModel


class RMQMessageDTO(BaseModel):
    """DTO for publishing message in RabbitMQ."""

    exchange_name: str
    routing_key: str
    # message: Optional[Union[str, bytes]]
    message: str
    queue_name: Optional[str]
