"""db.base"""

from ormar import ModelMeta

from fastapi_pytorch_postgresql_sandbox.db.config import database
from fastapi_pytorch_postgresql_sandbox.db.meta import meta


class BaseMeta(ModelMeta):
    """Base metadata for models."""

    database = database
    metadata = meta
