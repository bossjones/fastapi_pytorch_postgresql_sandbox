# pylint: disable=no-self-use
import os

import pytest

from fastapi_pytorch_postgresql_sandbox import settings
from fastapi_pytorch_postgresql_sandbox.utils.file_functions import tilda

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))


@pytest.mark.unittest
class TestSettings:
    def test_defaults(
        self,
    ) -> None:
        test_settings: settings.Settings = settings.Settings()

        assert test_settings.arch == "efficientnet_b0"
        assert test_settings.class_names == ["twitter", "facebook", "tiktok"]
        assert test_settings.db_base == "fastapi_pytorch_postgresql_sandbox_test"
        assert not test_settings.db_echo
        assert test_settings.db_host == "localhost"
        assert test_settings.db_pass == "fastapi_pytorch_postgresql_sandbox"
        assert test_settings.db_port == 5432
        assert test_settings.db_user == "fastapi_pytorch_postgresql_sandbox"
        assert test_settings.environment == "dev"
        assert not test_settings.gpu

        if not IS_RUNNING_ON_GITHUB_ACTIONS:
            assert test_settings.host == "localhost"
        else:
            assert test_settings.host == "127.0.0.1"

        assert test_settings.lr == 0.001
        assert test_settings.model_weights == "EfficientNet_B0_Weights"
        # assert not test_settings.opentelemetry_endpoint
        assert test_settings.port == 8008
        if not IS_RUNNING_ON_GITHUB_ACTIONS:
            assert str(test_settings.prometheus_dir) == "prom"
        assert test_settings.pytorch_device == "mps"
        assert test_settings.rabbit_channel_pool_size == 10
        assert test_settings.rabbit_host == "localhost"
        assert test_settings.rabbit_pass == "guest"
        assert test_settings.rabbit_pool_size == 2
        assert test_settings.rabbit_port == 5672
        assert test_settings.rabbit_user == "guest"
        assert test_settings.rabbit_vhost == "/"
        assert not test_settings.redis_base
        assert test_settings.redis_host == "localhost"
        assert not test_settings.redis_pass
        assert test_settings.redis_port == 6379
        assert not test_settings.redis_user
        if not IS_RUNNING_ON_GITHUB_ACTIONS:
            assert test_settings.reload
        assert test_settings.seed == 42
        assert test_settings.weights == tilda(
            "~/Documents/my_models/ScreenNetV1.pth",
        )
        assert test_settings.workers_count == 1

        assert test_settings.worker_exchange_name == "screenet"
        assert test_settings.worker_queue_name == "screennet_inference_queue"
        assert test_settings.worker_routing_key == "classify_worker"
