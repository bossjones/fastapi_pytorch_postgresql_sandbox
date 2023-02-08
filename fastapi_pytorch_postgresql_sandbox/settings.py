""" settings """
from __future__ import annotations

import enum
from pathlib import Path
from tempfile import gettempdir
from typing import List, Optional

from pydantic import BaseSettings
from yarl import URL

from fastapi_pytorch_postgresql_sandbox.deeplearning.architecture.screennet.config import (
    PATH_TO_BEST_MODEL,
)

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):  # noqa: WPS600
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO

    # Variables for the database
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "fastapi_pytorch_postgresql_sandbox"
    db_pass: str = "fastapi_pytorch_postgresql_sandbox"
    db_base: str = "fastapi_pytorch_postgresql_sandbox"
    db_echo: bool = False

    pytorch_device: str = "mps"
    arch: str = "efficientnet_b0"
    model_weigths: str = "EfficientNet_B0_Weights"
    class_names: List[str] = ["twitter", "facebook", "tiktok"]
    gpu: bool = False
    weights: str = PATH_TO_BEST_MODEL

    # Variables for Redis
    redis_host: str = "fastapi_pytorch_postgresql_sandbox-redis"
    redis_port: int = 6379
    redis_user: Optional[str] = None
    redis_pass: Optional[str] = None
    redis_base: Optional[int] = None

    # Variables for RabbitMQ
    rabbit_host: str = "fastapi_pytorch_postgresql_sandbox-rmq"
    rabbit_port: int = 5672
    rabbit_user: str = "guest"
    rabbit_pass: str = "guest"
    rabbit_vhost: str = "/"

    rabbit_pool_size: int = 2
    rabbit_channel_pool_size: int = 10

    # This variable is used to define
    # multiproc_dir. It's required for [uvi|guni]corn projects.
    prometheus_dir: Path = TEMP_DIR / "prom"

    # Grpc endpoint for opentelemetry.
    # E.G. http://localhost:4317
    opentelemetry_endpoint: Optional[str] = None

    # kafka_bootstrap_servers: list[str] = [
    #     "fastapi_pytorch_postgresql_sandbox-kafka:9092",
    # ]

    @property
    def db_url(self) -> URL:
        """
        Assemble database URL from settings.

        :return: database URL.
        """
        return URL.build(
            scheme="postgresql",
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_pass,
            path=f"/{self.db_base}",
        )

    @property
    def redis_url(self) -> URL:
        """
        Assemble REDIS URL from settings.

        :return: redis URL.
        """
        path = f"/{self.redis_base}" if self.redis_base is not None else ""
        return URL.build(
            scheme="redis",
            host=self.redis_host,
            port=self.redis_port,
            user=self.redis_user,
            password=self.redis_pass,
            path=path,
        )

    @property
    def rabbit_url(self) -> URL:
        """
        Assemble RabbitMQ URL from settings.

        :return: rabbit URL.
        """
        return URL.build(
            scheme="amqp",
            host=self.rabbit_host,
            port=self.rabbit_port,
            user=self.rabbit_user,
            password=self.rabbit_pass,
            path=self.rabbit_vhost,
        )

    class Config:
        env_file = ".env"
        env_prefix = "FASTAPI_PYTORCH_POSTGRESQL_SANDBOX_"
        env_file_encoding = "utf-8"


settings = Settings()
