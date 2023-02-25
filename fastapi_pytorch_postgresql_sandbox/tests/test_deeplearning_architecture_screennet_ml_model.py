# pylint: disable=no-self-use
import os

import pytest

from fastapi_pytorch_postgresql_sandbox import settings
from fastapi_pytorch_postgresql_sandbox.deeplearning.architecture.screennet import (
    ml_model,
)
from fastapi_pytorch_postgresql_sandbox.utils.file_functions import tilda

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))


@pytest.mark.xfail(
    reason="Looks like the model from state file is still busted, need to fix",
)
@pytest.mark.unittest
class TestImageClassifier:
    @pytest.mark.torchtests
    def test_load_model(
        self,
    ) -> None:
        test_model: ml_model.ImageClassifier = ml_model.ImageClassifier()

        assert test_model.path_to_model == tilda(
            "~/Documents/my_models/ScreenCropNetV1_378_epochs.pth",
        )

        assert str(test_model.device) == "mps"

        # import bpdb

        # bpdb.set_trace()

        test_model.load_model(pretrained=True)


@pytest.mark.xfail(
    reason="Looks like the model from state file is still busted, need to fix",
)
class TestMlModelFunctions:
    def test_create_effnetb0_model(self):
        test_settings = settings.Settings()
        test_effnetb0 = ml_model.create_effnetb0_model(
            "mps",
            ["twitter", "facebook", "tiktok"],
            test_settings,
        )

        test_device = next(test_effnetb0.parameters()).device
        assert str(test_device) == "mps:0"
