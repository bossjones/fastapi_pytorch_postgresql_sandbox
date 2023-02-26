# sourcery skip: no-complex-if-expressions
# pylint: disable=no-self-use
import os

import pytest
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from fastapi_pytorch_postgresql_sandbox import settings
from fastapi_pytorch_postgresql_sandbox.deeplearning.architecture.screennet import (
    ml_model,
)
from fastapi_pytorch_postgresql_sandbox.utils.file_functions import tilda

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))


class TestImageClassifier:
    @pytest.mark.torchtests
    def test_load_model(
        self,
    ) -> None:
        test_model: ml_model.ImageClassifier = ml_model.ImageClassifier()

        assert test_model.path_to_model == tilda(
            "~/Documents/my_models/ScreenNetV1.pth",
        )

        if not IS_RUNNING_ON_GITHUB_ACTIONS:
            assert str(test_model.device) == "mps"

        # import bpdb

        # bpdb.set_trace()

        test_model.load_model(pretrained=True)

        assert type(test_model.loss_fn) == CrossEntropyLoss
        assert type(test_model.optimizer) == Adam
        assert test_model.class_names == ["twitter", "facebook", "tiktok"]
        assert test_model.path_to_model == tilda(
            "~/Documents/my_models/ScreenNetV1.pth",
        )


# @pytest.mark.xfail(
#     reason="Looks like the model from state file is still busted, need to fix",
# )
class TestMlModelFunctions:
    def test_create_effnetb0_model(self):
        test_settings = settings.Settings()
        test_effnetb0 = (
            ml_model.create_effnetb0_model(
                torch.device("cpu"),
                ["twitter", "facebook", "tiktok"],
                test_settings,
            )
            if IS_RUNNING_ON_GITHUB_ACTIONS
            else ml_model.create_effnetb0_model(
                torch.device("mps"),
                ["twitter", "facebook", "tiktok"],
                test_settings,
            )
        )
        test_device = next(test_effnetb0.parameters()).device
        assert str(test_device) == "mps:0"
