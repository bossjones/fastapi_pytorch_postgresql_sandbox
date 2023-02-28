# type: ignore
"""ml_client"""
from fastapi_pytorch_postgresql_sandbox.ml_client._version import __version__
from fastapi_pytorch_postgresql_sandbox.ml_client.client import (
    ApifyClient,
    ApifyClientAsync,
)

__all__ = ["ApifyClient", "ApifyClientAsync", "__version__"]
