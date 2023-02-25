import mlxtend
import pytest
import torch
import torchvision


@pytest.mark.integration
class TestVersions:
    @pytest.mark.torchtests
    def test_versions(
        self,
    ) -> None:
        assert (
            int(torch.__version__.split(".")[1]) >= 12
        ), "torch version should be 1.12+"
        assert (
            int(torchvision.__version__.split(".")[1]) >= 13
        ), "torchvision version should be 0.13+"
        assert (
            int(mlxtend.__version__.split(".")[1]) >= 19
        ), "mlxtend verison should be 0.19.0 or higher"
