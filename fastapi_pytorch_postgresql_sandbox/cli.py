"""cli"""
# sourcery skip: avoid-global-variables
# pylint: disable=no-name-in-module
# pylint: disable=no-member

# NOTE: For more examples tqdm + aiofile, search https://github.com/search?l=Python&q=aiofile+tqdm&type=Code
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
from mimetypes import MimeTypes
import time
from typing import Any, List

import aiometer
from codetiming import Timer
import httpx
import rich
from rich.pretty import pprint

from fastapi_pytorch_postgresql_sandbox.utils.file_functions import (
    PathLike,
    go_get_image_files,
)

# from tqdm.asyncio import tqdm


mime = MimeTypes()

session = httpx.AsyncClient()


# >>> files = {'upload-file': ('report.xls', open('report.xls', 'rb'), 'application/vnd.ms-excel')}
# >>> r = httpx.post("https://httpbin.org/post", files=files)
# >>> print(r.text)
# {
#   ...
#   "files": {
#     "upload-file": "<... binary content ...>"
#   },
#   ...

# }


def run_inspect(obj: Any):
    rich.inspect(obj, all=True)


async def fetch(client: httpx.AsyncClient, request: httpx.Request):
    import bpdb

    response = await client.send(request)
    bpdb.set_trace()
    return response.json()


# async def send_to_classify_api(url: PathLike, client: Any):
#     """_summary_

#     Args:
#         url (PathLike): _description_

#     Returns:
#         _type_: _description_
#     """
#     mime_type: tuple[str | None, str | None] = mime.guess_type(f"{url}")
#     import bpdb

#     # files = {"upload-file": (f"{url}", open(f"{url}", "rb"), mime_type[0])}
#     files = {"image": (f"{url}", open(f"{url}", "rb"), f"{mime_type[0]}")}
#     # headers = headers = {
#     #     "accept": "application/json",
#     #     # "content-type": "multipart/form-data; charset=utf-8",
#     # }
#     response = await client.post(
#         "http://localhost:8008/api/screennet/classify",
#         # headers=headers,
#         files=files,
#     )
#     bpdb.set_trace()

#     # response = await session.get(url)
#     return response.json()


async def go_partial(loop: Any) -> List[PathLike]:
    """entrypoint

    Args:
        loop (_type_): _description_

    Returns:
        _type_: _description_
    """
    path_to_dir = "/Users/malcolm/Downloads/datasets/twitter_facebook_tiktok/"

    handle_go_get_image_files_func = functools.partial(go_get_image_files, path_to_dir)

    with Timer(text="\nTotal elapsed time: {:.1f}"):
        # 2. Run in a custom thread pool:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            images = await loop.run_in_executor(pool, handle_go_get_image_files_func)

            # pprint(images)
            print(f"Found {len(images)} images")
    # urls = ["http://httpbin.org/html" for i in range(10)]
    # with Timer(text="\nTotal elapsed time: {:.1f}"):
    #     results = await aiometer.run_on_each(
    #         send_to_classify_api,
    #         images,
    #         # max_at_once=10, # Limit maximum number of concurrently running tasks.
    #         max_per_second=1,  # here we can set max rate per second
    #     )

    # ---------------------------------------------------------
    # chunked_lists = list(misc.divide_chunks(file_to_upload, n=10))
    # discord has a limit of 10 media uploads per api call. break them up.
    # SOURCE: https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    num = 2
    final = [
        images[i * num : (i + 1) * num] for i in range((len(images) + num - 1) // num)
    ]

    for count, chunk in enumerate(final):
        print(f"count = {count}")
        requests = []
        for img in chunk:
            mime_type: tuple[str | None, str | None] = mime.guess_type(f"{img}")
            # files = {"file": (None, open(f"{img}", "rb"), f"{mime_type[0]}")}
            files = {"upload-file": (None, open(f"{img}", "rb"), f"{mime_type[0]}")}
            headers = headers = {
                "accept": "application/json",
            }
            data = {"file": f"{img}", "type": f"{mime_type[0]}"}
            api_request = httpx.Request(
                "POST",
                "http://localhost:8008/api/screennet/classify",
                files=files,
                headers=headers,
                data=data,
            )
            _ = await api_request.aread()
            requests.append(api_request)

        # import bpdb

        # bpdb.set_trace()
        # pprint(requests)
        # await aiometer.run_all()
        # jobs = [functools.partial(send_to_classify_api, image, session) for image in images]
        jobs = [functools.partial(fetch, session, request) for request in requests]
        results = await aiometer.run_all(jobs, max_at_once=10, max_per_second=5)
        rich.print(" -> results: \n")
        pprint(results)

    return images


if __name__ == "__main__":
    session = httpx.AsyncClient()
    start_time = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(go_partial(loop))
    duration = time.time() - start_time
    print(f"Computed in {duration} seconds")
