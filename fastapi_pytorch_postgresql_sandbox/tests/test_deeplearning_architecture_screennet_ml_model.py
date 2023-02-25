# pylint: disable=no-self-use
import os

import pytest

from fastapi_pytorch_postgresql_sandbox.deeplearning.architecture.screennet import (
    ml_model,
)
from fastapi_pytorch_postgresql_sandbox.utils.file_functions import tilda

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))


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
