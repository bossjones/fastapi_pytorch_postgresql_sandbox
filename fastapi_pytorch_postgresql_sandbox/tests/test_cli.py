# pylint: disable=no-self-use
# NOTE: Testing
# NOTE: https://til.simonwillison.net/pytest/pytest-argparse
# NOTE: https://realpython.com/async-io-python/
import argparse
import asyncio
import contextlib
import os
import pathlib
import uuid

import pytest

from fastapi_pytorch_postgresql_sandbox import cli
from fastapi_pytorch_postgresql_sandbox.utils.file_functions import fix_path

HERE = os.path.dirname(__file__)

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))


# ['--predict', './fastapi_pytorch_postgresql_sandbox/tests/fixtures']
@pytest.mark.unittest
class TestCLI:
    @pytest.mark.parametrize("option", ("-h", "--help"))
    def test_cli_help(self, capsys, option) -> None:
        # with patch("sys.argv", ["main", "--name", "Jürgen"]):
        with contextlib.suppress(SystemExit):
            _: argparse.Namespace = cli.main(
                [option],
            )  # sourcery skip: avoid-single-character-names-variables
        output = capsys.readouterr().out
        assert "Screennet cli tool" in output

    def test_get_chunked_lists(self) -> None:
        paths = [pathlib.Path(f"{uuid.uuid4().hex}") for _ in range(100)]

        actual = cli.get_chunked_lists(paths, 2)  # type: ignore

        assert len(actual) == 50


# @pytest.mark.anyio
@pytest.mark.asyncio
class TestCLIAsync:
    async def test_aio_get_images(self):
        loop: asyncio.AbstractEventLoop = (
            asyncio.get_event_loop_policy().get_event_loop()
        )  # type: ignore
        test_path_dir = fix_path(f"{HERE}/fixtures")
        images = await cli.aio_get_images(loop, test_path_dir)

        assert len(images) == 1
