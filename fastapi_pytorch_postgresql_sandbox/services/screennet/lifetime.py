"""services.screennet.lifetime"""

from fastapi import FastAPI

from fastapi_pytorch_postgresql_sandbox.deeplearning.architecture.screennet.ml_model import (
    ImageClassifier,
)


def init_screennet(app: FastAPI) -> None:  # pragma: no cover
    """
    Creates screennet model.

    :param app: current fastapi application.
    """
    net_api = ImageClassifier()
    net_api.load_model()

    app.state.net = net_api


# async def shutdown_redis(app: FastAPI) -> None:  # pragma: no cover
#     """
#     Closes redis connection pool.

#     :param app: current FastAPI app.
#     """
#     await app.state.redis_pool.disconnect()
