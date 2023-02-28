# pylint: disable=no-self-use
# NOTE: Testing
# NOTE: https://til.simonwillison.net/pytest/pytest-argparse
# NOTE: https://realpython.com/async-io-python/
import argparse
import os

import pytest

from fastapi_pytorch_postgresql_sandbox import cli
from fastapi_pytorch_postgresql_sandbox.utils.file_functions import tilda

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))
# ['--predict', './fastapi_pytorch_postgresql_sandbox/tests/fixtures']
@pytest.mark.parametrize("option", ("-h", "--help"))
@pytest.mark.unittest
class TestSettings:
    def test_defaults(self, capsys, option) -> None:
        # with patch("sys.argv", ["main", "--name", "Jürgen"]):
        try:
            test_cli_parsed: argparse.Namespace = cli.main([option])
        except SystemExit:
            pass
        output = capsys.readouterr().out
        assert "Screennet cli tool" in output
