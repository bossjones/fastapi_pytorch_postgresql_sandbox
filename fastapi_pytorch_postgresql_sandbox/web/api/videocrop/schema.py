# pylint: disable=no-name-in-module
"""web.api.screennet"""

from typing import Dict, Union

from pydantic import BaseModel


class PendingCropDTO(BaseModel):
    """Data Transfer Object(DTO) for pending crop values."""

    inference_id: str


class CropResultDTO(BaseModel):
    """Data Transfer Object(DTO) for result crop values."""

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
