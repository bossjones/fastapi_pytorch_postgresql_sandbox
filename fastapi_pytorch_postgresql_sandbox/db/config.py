from databases import Database

from fastapi_pytorch_postgresql_sandbox.settings import settings

database = Database(str(settings.db_url))
