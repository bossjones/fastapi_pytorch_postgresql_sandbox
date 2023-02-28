"""cli"""
# sourcery skip: avoid-global-variables
# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=consider-using-with

# NOTE: For more examples tqdm + aiofile, search https://github.com/search?l=Python&q=aiofile+tqdm&type=Code
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
from mimetypes import MimeTypes
import time
from typing import Any, Dict, List

import aiometer
from codetiming import Timer
import httpx
from icecream import ic
import rich

from fastapi_pytorch_postgresql_sandbox.utils.file_functions import (
    PathLike,
    go_get_image_files,
)

JSONType = str | int | float | bool | None | Dict | List

WORKERS = 100

mime = MimeTypes()

session = httpx.AsyncClient()


def run_inspect(obj: Any) -> None:
    """_summary_

    Args:
        obj (Any): _description_
    """
    rich.inspect(obj, all=True)


async def fetch(client: httpx.AsyncClient, request: httpx.Request) -> JSONType:
    """_summary_

    Args:
        client (httpx.AsyncClient): _description_
        request (httpx.Request): _description_

    Returns:
        _type_: _description_
    """
    response: httpx.Response = await client.send(request)
    return response.json()


async def aio_get_images() -> List[PathLike]:
    """Get all images inside of a directory

    Returns:
        _type_: _description_
    """
    path_to_dir = "/Users/malcolm/Downloads/datasets/twitter_facebook_tiktok/"

    handle_go_get_image_files_func = functools.partial(go_get_image_files, path_to_dir)

    with Timer(text="\nTotal elapsed time: {:.1f}"):
        # 2. Run in a custom thread pool:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            images = await loop.run_in_executor(pool, handle_go_get_image_files_func)

            print(f"Found {len(images)} images")

    return images


def get_chunked_lists(
    img_paths: List[PathLike], num: int = WORKERS,
) -> list[list[PathLike]]:
    """Chunked_lists = list(misc.divide_chunks(file_to_upload, n=10)).
    discord has a limit of 10 media uploads per api call. break them up.

    Args:
        img_paths (List[PathLike]): _description_
        num (int, optional): _description_. Defaults to WORKERS.

    Returns:
        list[list[PathLike]]: _description_
    """
    # ---------------------------------------------------------
    # chunked_lists = list(misc.divide_chunks(file_to_upload, n=10))
    # discord has a limit of 10 media uploads per api call. break them up.
    # SOURCE: https://www.geeksforgeeks.org/break-list-chunks-size-n-python/

    return [
        img_paths[i * num : (i + 1) * num]
        for i in range((len(img_paths) + num - 1) // num)
    ]


async def go_partial(loop: Any) -> List[PathLike]:
    """entrypoint

    Args:
        loop (_type_): _description_

    Returns:
        _type_: _description_
    """
    images = await aio_get_images()

    # ---------------------------------------------------------
    # chunked_lists = list(misc.divide_chunks(file_to_upload, n=10))
    # discord has a limit of 10 media uploads per api call. break them up.
    # SOURCE: https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    final = get_chunked_lists(images)

    for count, chunk in enumerate(final):
        print(f"count = {count}")
        requests = []
        for img in chunk:
            mime_type: tuple[str | None, str | None] = mime.guess_type(f"{img}")
            files = {"file": (img.name, open(f"{img}", "rb"), f"{mime_type[0]}")}  # type: ignore
            headers = headers = {
                "accept": "application/json",
            }

            data = {"type": f"{mime_type[0]}"}
            api_request = httpx.Request(
                "POST",
                "http://localhost:8008/api/screennet/classify",
                files=files,
                headers=headers,
                data=data,
            )
            _ = await api_request.aread()
            requests.append(api_request)

        jobs = [functools.partial(fetch, session, request) for request in requests]

        results = await aiometer.run_all(
            jobs,
            max_at_once=WORKERS,
            max_per_second=WORKERS,
        )

        ic(results)

    return images


if __name__ == "__main__":
    session = httpx.AsyncClient()
    start_time = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(go_partial(loop))
    duration = time.time() - start_time
    print(f"Computed in {duration} seconds")
