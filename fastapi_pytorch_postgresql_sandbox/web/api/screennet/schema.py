# pylint: disable=no-name-in-module
"""web.api.screennet"""

from typing import Dict, Union

from pydantic import BaseModel


class PendingClassificationDTO(BaseModel):
    """Data Transfer Object(DTO) for pending classification values."""

    inference_id: str


class ClassificationResultDTO(BaseModel):
    """Data Transfer Object(DTO) for result classification values."""

    predicted_class: int


class RedisPredictionData(BaseModel):
    """Represents dictory data set inside of Redis"""

    pred_prob: float
    pred_class: str
    time_for_pred: float


class RedisPredictionValueDTO(BaseModel):
    """Data Transfer Object(DTO) for classify prediction redis values."""

    # data: Dict[str, RedisPredictionData]
    data: Dict[str, Union[str, Dict[str, Dict[str, Union[float, str]]]]]
