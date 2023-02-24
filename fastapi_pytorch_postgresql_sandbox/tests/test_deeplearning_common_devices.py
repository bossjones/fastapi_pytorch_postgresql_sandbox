import os

import pytest

from fastapi_pytorch_postgresql_sandbox.deeplearning.common import devices

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))


@pytest.mark.unittest
class TestMeilisearchCaseIsolated:
    @pytest.mark.skipif(
        IS_RUNNING_ON_GITHUB_ACTIONS,
        reason="Not sure if mps is enabled on github actions yet, disabling",
    )
    @pytest.mark.torchtests
    def test_has_mps(
        self,
    ) -> None:
        assert devices.has_mps()
