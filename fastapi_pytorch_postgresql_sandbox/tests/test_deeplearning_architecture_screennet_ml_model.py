# sourcery skip: no-complex-if-expressions
# pylint: disable=no-self-use
from io import BytesIO
import os
import pathlib

from PIL import Image
import pytest
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from fastapi_pytorch_postgresql_sandbox import settings
from fastapi_pytorch_postgresql_sandbox.deeplearning.architecture.screennet import (
    ml_model,
)
from fastapi_pytorch_postgresql_sandbox.utils.file_functions import tilda
from fastapi_pytorch_postgresql_sandbox.utils.mlops import (
    convert_pil_image_to_rgb_channels,
)

HERE = os.path.dirname(__file__)

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
        assert test_model.class_names == ["facebook", "tiktok", "twitter"]
        assert test_model.path_to_model == tilda(
            "~/Documents/my_models/ScreenNetV1.pth",
        )

    @pytest.mark.torchtests
    def test_infer_from_pathlib(self):
        """Test pathlib implementation of model running"""
        # init model
        test_model: ml_model.ImageClassifier = ml_model.ImageClassifier()
        test_model.load_model(pretrained=True)

        # setup path api
        path = pathlib.Path(
            f"{HERE}/fixtures/test1.jpg",
        )

        paths = [path]

        image_data = convert_pil_image_to_rgb_channels(f"{paths[0]}")

        test_predict = test_model.infer(image_data)

        image_class = test_predict[0]["pred_class"]
        image_pred_prob = test_predict[0]["pred_prob"]
        image_time_for_pred = test_predict[0]["time_for_pred"]

        # 5. Print metadata
        print(f"Random image path: {paths[0]}")
        print(f"Image class: {image_class}")
        print(f"Image pred prob: {image_pred_prob}")
        print(f"Image pred time: {image_time_for_pred}")
        print(f"Image height: {image_data.height}")
        print(f"Image width: {image_data.width}")

        assert image_class == "twitter"

    @pytest.mark.torchtests
    def test_infer_from_bytesio(self):
        """_summary_"""
        # init model
        test_model: ml_model.ImageClassifier = ml_model.ImageClassifier()
        test_model.load_model(pretrained=True)

        # setup path api
        path = pathlib.Path(
            f"{HERE}/fixtures/test1.jpg",
        )
        with path.open("rb") as request_object_content:
            content = request_object_content.read()
            pil_image: Image = Image.open(BytesIO(content))
            image_data: Image = convert_pil_image_to_rgb_channels(f"{path}")
            # assert image_data.format == "JPEG"


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
