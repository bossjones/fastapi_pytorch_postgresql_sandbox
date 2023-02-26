# pylint: disable=no-name-in-module
"""web.api.redis"""
from typing import Dict, Optional, Union

from pydantic import BaseModel


class RedisValueDTO(BaseModel):
    """Data Transfer Object(DTO) for redis values."""

    key: str
    value: Optional[str]  # noqa: WPS110


class RedisPredictionData(BaseModel):
    """Represents dictory data set inside of Redis"""

    pred_prob: float
    pred_class: str
    time_for_pred: float


class RedisPredictionValueDTO(BaseModel):
    """Data Transfer Object(DTO) for classify prediction redis values."""

    # data: Dict[str, RedisPredictionData]
    data: Dict[str, Union[str, Dict[str, Dict[str, Union[float, str]]]]]
