from fastapi.routing import APIRouter

from fastapi_pytorch_postgresql_sandbox.web.api import (  # kafka,
    docs,
    dummy,
    echo,
    monitoring,
    rabbit,
    redis,
    screennet,
)

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(docs.router)
api_router.include_router(echo.router, prefix="/echo", tags=["echo"])
api_router.include_router(dummy.router, prefix="/dummy", tags=["dummy"])
api_router.include_router(redis.router, prefix="/redis", tags=["redis"])
api_router.include_router(rabbit.router, prefix="/rabbit", tags=["rabbit"])
api_router.include_router(screennet.router, prefix="/screennet", tags=["screennet"])
# api_router.include_router(kafka.router, prefix="/kafka", tags=["kafka"])
