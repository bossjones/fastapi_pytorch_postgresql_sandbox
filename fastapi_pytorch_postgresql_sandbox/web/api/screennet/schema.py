# pylint: disable=no-name-in-module
"""web.api.screennet"""

from pydantic import BaseModel


class PendingClassificationDTO(BaseModel):
    """Data Transfer Object(DTO) for pending classification values."""

    inference_id: str


class ClassificationResultDTO(BaseModel):
    """Data Transfer Object(DTO) for result classification values."""

    predicted_class: int
