# pylint: disable=no-name-in-module
"""web.api.redis"""
from typing import Optional

from pydantic import BaseModel


class RedisValueDTO(BaseModel):
    """Data Transfer Object(DTO) for redis values."""

    key: str
    value: Optional[str]  # noqa: WPS110
