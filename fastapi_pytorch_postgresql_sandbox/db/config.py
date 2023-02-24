# sourcery skip: avoid-global-variables
""" db.config """
from __future__ import annotations

from databases import Database

from fastapi_pytorch_postgresql_sandbox.settings import settings

database = Database(str(settings.db_url))
