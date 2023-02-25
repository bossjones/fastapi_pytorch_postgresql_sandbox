"""services.screennet.dependency"""
# # sourcery skip: avoid-global-variables
# # pylint: disable=no-name-in-module
# from typing import AsyncGenerator

# from starlette.requests import Request

# from fastapi_pytorch_postgresql_sandbox.deeplearning.architecture.screennet.ml_model import (
#     ImageClassifier,
# )


# async def get_screennet(
#     request: Request,
# ) -> AsyncGenerator[ImageClassifier, None]:  # pragma: no cover
#     """
#     Returns connection pool.

#     You can use it like this:

#     >>> from redis.asyncio import ConnectionPool, Redis
#     >>>
#     >>> async def handler(redis_pool: ConnectionPool = Depends(get_redis_pool)):
#     >>>     async with Redis(connection_pool=redis_pool) as redis:
#     >>>         await redis.get('key')

#     I use pools so you don't acquire connection till the end of the handler.

#     :param request: current request.
#     :returns:  redis connection pool.
#     """
#     return request.app.state.redis_pool
