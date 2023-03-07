"""cli"""
# sourcery skip: avoid-global-variables
# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=consider-using-with

# NOTE: For more examples tqdm + aiofile, search https://github.com/search?l=Python&q=aiofile+tqdm&type=Code
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
from datetime import datetime
import functools
import gc
import json
from mimetypes import MimeTypes
import os
import sys
import time
from typing import Any, Dict, List, Union

from aiocsv import AsyncDictReader, AsyncDictWriter, AsyncReader, AsyncWriter
import aiofiles
import aiometer
from codetiming import Timer
import httpx
from httpx import Request, Response
from icecream import ic
from pydantic import BaseModel, Field
import rich
import tenacity
from tenacity import TryAgain

from fastapi_pytorch_postgresql_sandbox import retry
from fastapi_pytorch_postgresql_sandbox.utils.file_functions import (
    PathLike,
    go_get_image_files,
)

JSONType = str | int | float | bool | None | Dict | List

WORKERS = 100

mime = MimeTypes()

# session = httpx.AsyncClient()

DEFAULT_PATH_TO_DIR = "/Users/malcolm/Downloads/datasets/twitter_facebook_tiktok/"


def convert_datetime_to_iso_8601_with_z_suffix(dt: datetime) -> str:
    """convert dattime object to string format

    Args:
        dt (datetime): _description_

    Returns:
        str: _description_
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class PredictionDataRow(BaseModel):
    """Dataclass for line that will be written to a CSV file

    Args:
        BaseModel (_type_): _description_
    """

    file_name: str
    # inference_id: str
    classifyed_pred_prob: Union[float, str]
    pred_prob_pred_class: str
    pred_prob_time_for_pred: Union[float, str]
    ts: datetime = Field(default_factory=datetime.now)

    # SOURCE: https://stackoverflow.com/questions/66548586/how-to-change-date-format-in-pydantic
    class Config:
        json_encoders = {
            # custom output conversion for datetime
            datetime: convert_datetime_to_iso_8601_with_z_suffix,
        }


class MemSafeFileInfo(BaseModel):
    """Data Container for all the info we need to lookup the specific values later

    Args:
        FileInfo (_type_): _description_
    """

    path_to_file: str
    request: Request

    class Config:
        arbitrary_types_allowed = True


class FileInfo(BaseModel):
    """Data Container for all the info we need to lookup the specific values later

    Args:
        FileInfo (_type_): _description_
    """

    path_to_file: str
    request: Request

    class Config:
        arbitrary_types_allowed = True


class FileInfoDTO(FileInfo):
    """_summary_

    Args:
        FileInfo (_type_): _description_
    """

    response: Response


def run_inspect(obj: Any) -> None:
    """_summary_

    Args:
        obj (Any): _description_
    """
    rich.inspect(obj, all=True)


class InputApiClassifyData(BaseModel):
    """Data Container for all the info we need to lookup the specific values later

    Args:
        FileInfo (_type_): _description_
    """

    path_to_file: str
    request: Request

    class Config:
        arbitrary_types_allowed = True


class OutputApiClassifyData(BaseModel):
    """Data Container for all the info we need to lookup the specific values later

    Args:
        FileInfo (_type_): _description_
    """

    path_to_file: str
    inference_id: str


class InputGetPredictionData(BaseModel):
    """Data Container for all the info we need to lookup the specific values later

    Args:
        FileInfo (_type_): _description_
    """

    path_to_file: str
    request: Request

    class Config:
        arbitrary_types_allowed = True


async def api_request_prediction(
    client: httpx.AsyncClient,
    file_info: Union[FileInfo, InputApiClassifyData],
) -> Union[FileInfoDTO, OutputApiClassifyData]:
    """Peform POST request to classify a given image against our api.

    Args:
        client (httpx.AsyncClient): _description_
        request (httpx.Request): _description_

    Returns:
        _type_: _description_
    """

    await file_info.request.aread()

    response: httpx.Response = await client.send(file_info.request)

    # free up file descriptors
    # await response.aclose()
    await file_info.request.stream.aclose()  # type: ignore

    inference_id: str = response.json()["inference_id"]

    # return FileInfoDTO(
    #     path_to_file=file_info.path_to_file,
    #     request=file_info.request,
    #     response=response,
    # )
    return OutputApiClassifyData(
        path_to_file=f"{file_info.path_to_file}",
        inference_id=inference_id,
    )


@tenacity.retry(
    **retry.linear_backoff(
        wait=tenacity.wait_fixed(3),
        stop=tenacity.stop_after_attempt(10),
    )
)
async def api_get_prediction_results(
    client: httpx.AsyncClient,
    file_info_dto: Union[FileInfoDTO, InputGetPredictionData],
) -> PredictionDataRow:
    """Peform GET request to pull back classify results for an image against our api.

    Args:
        client (httpx.AsyncClient): _description_
        request (httpx.Request): _description_

    Returns:
        _type_: _description_
    """
    response: httpx.Response = await client.send(file_info_dto.request)

    # import bpdb

    # bpdb.set_trace()
    # Prediction results are not ready yet, retry
    if response.status_code == 202:
        raise TryAgain
    pred_data_row = PredictionDataRow(
        file_name=f"{file_info_dto.path_to_file}",
        classifyed_pred_prob=float(response.json()["data"]["pred_prob"]),
        pred_prob_pred_class=response.json()["data"]["pred_class"],
        pred_prob_time_for_pred=float(response.json()["data"]["time_for_pred"]),
    )

    ic(pred_data_row)

    # free up file descriptor
    await file_info_dto.request.stream.aclose()  # type: ignore
    await response.aclose()
    # await

    return pred_data_row


async def aio_run_api_classify(
    completed: list,
    final: list[list[PathLike]],
    workers: int,
) -> Union[List[List[FileInfoDTO]], List[List[OutputApiClassifyData]]]:
    """wrapper function to peform classify request

    Args:
        completed (list): _description_
        final (list[list[PathLike]]): _description_

    Returns:
        List[List[FileInfoDTO]]: _description_
    """
    # send post request to perform prediction
    for count, chunk in enumerate(final):
        print(f"[aio_run_api_classify] count = {count}")
        # requests = []
        file_infos: list = []
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

            # NOTE: ORIG # a_file_info = FileInfo(path_to_file=f"{img}", request=api_request)
            a_file_info: InputApiClassifyData = InputApiClassifyData(
                path_to_file=f"{img}",
                request=api_request,
            )
            file_infos.append(a_file_info)
            # gc.collect()

        # jobs = [functools.partial(fetch, session, request) for request in requests]
        jobs = [
            functools.partial(api_request_prediction, session, fi_obj)
            for fi_obj in file_infos
        ]

        results = await aiometer.run_all(
            jobs,
            max_at_once=workers,
            max_per_second=workers,
        )

        completed.append(results)

    return completed


async def aio_run_api_get_classify_results(
    completed: list,
    final: Union[List[List[FileInfoDTO]], List[List[OutputApiClassifyData]]],
    workers: int,
) -> List[List[PredictionDataRow]]:
    """wrapper function to peform classify request

    Args:
        completed (list): _description_
        final (list[list[PathLike]]): _description_

    Returns:
        List[List[PredictionDataRow]]: _description_
    """
    # send post request to perform prediction
    for count, chunk in enumerate(final):
        rich.print("aio_run_api_get_classify_results")
        ic(chunk)
        print(f"[aio_run_api_get_classify_results] count = {count}")
        updated_file_info_dtos: list = []
        for _fi_dto in chunk:
            headers = headers = {
                "accept": "application/json",
            }
            # inference_id = _fi_dto.response.json()["inference_id"]

            api_request = httpx.Request(
                "GET",
                f"http://localhost:8008/api/screennet/result/{_fi_dto.inference_id}",
                headers=headers,
            )
            # a_file_info_dto = FileInfoDTO(
            #     path_to_file=f"{_fi_dto.path_to_file}",
            #     request=api_request,
            #     response=_fi_dto.response,
            # )
            a_file_info_dto = InputGetPredictionData(
                path_to_file=f"{_fi_dto.path_to_file}",
                request=api_request,
            )
            updated_file_info_dtos.append(a_file_info_dto)

        jobs = [
            functools.partial(api_get_prediction_results, session, fi_dto_obj)
            for fi_dto_obj in updated_file_info_dtos
        ]

        results = await aiometer.run_all(
            jobs,
            max_at_once=workers,
            max_per_second=workers,
        )

        completed.append(results)

    return completed


async def aio_get_images(
    loop: asyncio.AbstractEventLoop,
    path_to_dir: str = DEFAULT_PATH_TO_DIR,
) -> List[PathLike]:
    """Get all images inside of a directory

    Returns:
        _type_: _description_
    """

    handle_go_get_image_files_func = functools.partial(go_get_image_files, path_to_dir)

    with Timer(text="\n [aio_get_images] Total elapsed time: {:.5f}"):
        # 2. Run in a custom thread pool:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            images = await loop.run_in_executor(pool, handle_go_get_image_files_func)

            print(f"Found {len(images)} images")

    return images


def get_chunked_lists(
    img_paths: List[PathLike],
    num: int = WORKERS,
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


def run_seralize_prediction_data_row(pdr: PredictionDataRow) -> JSONType:
    """Prepare Prediction Data Row for storage in csv

    Args:
        pdr (PredictionDataRow): _description_

    Returns:
        _type_: _description_
    """
    to_jsons = pdr.json()
    return json.loads(to_jsons)


async def aio_write_csv(path_to_csv: PathLike, prediction_data_rows: List[JSONType]):
    """_summary_

    Args:
        path_to_csv (PathLike): _description_
        prediction_data_rows (List[JSONType]): _description_
    """

    if not os.path.isfile(path_to_csv):  # type: ignore
        # dict writing, all quoted, "NULL" for missing fields
        async with aiofiles.open(
            path_to_csv,
            mode="w",
            encoding="utf-8",
            newline="",
        ) as afp:  # type: ignore
            writer = AsyncDictWriter(
                afp,
                [
                    "file_name",
                    "classifyed_pred_prob",
                    "pred_prob_pred_class",
                    "pred_prob_time_for_pred",
                    "ts",
                ],
                restval="NULL",
            )
            await writer.writeheader()
            await writer.writerows(prediction_data_rows)  # type: ignore
    else:
        # dict writing, all quoted, "NULL" for missing fields
        async with aiofiles.open(
            path_to_csv,
            mode="a",
            encoding="utf-8",
            newline="",
        ) as afp:  # type: ignore
            writer = AsyncDictWriter(
                afp,
                [
                    "file_name",
                    "classifyed_pred_prob",
                    "pred_prob_pred_class",
                    "pred_prob_time_for_pred",
                    "ts",
                ],
                restval="NULL",
            )
            await writer.writerows(prediction_data_rows)  # type: ignore


async def go_partial(
    loop: asyncio.AbstractEventLoop,
    args: argparse.Namespace,
) -> List[PathLike]:
    """entrypoint

    Args:
        loop (_type_): _description_

    Returns:
        _type_: _description_
    """
    images = await aio_get_images(loop, args.predict)

    file_info_dtos = []  # type: ignore
    prediction_data_rows = []  # type: ignore
    # ---------------------------------------------------------
    # chunked_lists = list(misc.divide_chunks(file_to_upload, n=10))
    # discord has a limit of 10 media uploads per api call. break them up.
    # SOURCE: https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    final: list[list[PathLike]] = get_chunked_lists(images)

    # Ask api to perform classify actions
    # completed_file_info_dtos: List[List[FileInfoDTO]] = await aio_run_api_classify(
    #     file_info_dtos,
    #     final,
    #     args.workers,
    # )
    completed_file_info_dtos: Union[
        List[List[FileInfoDTO]],
        List[List[OutputApiClassifyData]],
    ] = await aio_run_api_classify(
        file_info_dtos,
        final,
        args.workers,
    )

    # Get the prediction results back
    completed_prediction_data_rows: List[
        List[PredictionDataRow]
    ] = await aio_run_api_get_classify_results(
        prediction_data_rows,
        completed_file_info_dtos,
        args.workers,
    )

    await session.aclose()

    # [[PredictionDataRow(file_name='fastapi_pytorch_postgresql_sandbox/tests/fixtures/test1.jpg', classifyed_pred_prob=0.6357, pred_prob_pred_class='twitter', pred_prob_time_for_pred=0.2469, ts=datetime.datetime(2023, 3, 2, 13, 24, 55, 760148))]]

    ic(completed_prediction_data_rows)

    # How to flatten list in Python?
    flat_completed = [
        element for sublist in completed_prediction_data_rows for element in sublist
    ]

    seralized_prediction_data_rows = [
        run_seralize_prediction_data_row(pdr) for pdr in flat_completed
    ]

    await aio_write_csv("./test.csv", seralized_prediction_data_rows)

    # print(flat_completed)

    # completed List[List[FileInfoDTO]]= [[FileInfoDTO(path_to_file='fastapi_pytorch_postgresql_sandbox/tests/fixtures/test1.jpg', request=<Request('POST', 'http://localhost:8008/api/screennet/classify')>, response=<Response [202 Accepted]>)]]
    # {'inference_id': '29c5b85d-fbfa-4654-bf73-9e142f95a3c4'}
    # inference_id =  completed[0][0].response.json()["inference_id"]
    # path_to_file = completed[0][0].path_to_file
    # prediction
    # curl -X 'GET' \
    #   'http://localhost:8008/api/screennet/result/29c5b85d-fbfa-4654-bf73-9e142f95a3c4' \
    #   -H 'accept: application/json'
    # {
    #   "data": {
    #     "pred_prob": "0.6357",
    #     "pred_class": "twitter",
    #     "time_for_pred": "0.3787"
    #   }
    # }

    return images


parser = argparse.ArgumentParser(description="Screennet cli tool")
parser.add_argument(
    "--predict",
    default=DEFAULT_PATH_TO_DIR,
    type=str,
    metavar="PREDICT_PATH",
    help="path to image to run prediction on (default: none)",
)
parser.add_argument(
    "--workers",
    default=WORKERS,
    type=int,
    metavar="NUM_WORKERS",
    help="Number of workers to use (default: WORKERS)",
)


def main(args: Union[None, Any] = None) -> argparse.Namespace:
    """As you can see, main() takes an optional list of arguments. The default for that is None which will cause argparse to read sys.argv - but I can inject arguments to the function from my tests if I need to.

    Args:
        args (Union[None, Any], optional): _description_. Defaults to None.

    Returns:
        argparse.Namespace: _description_
    """
    ic(args)
    ic(type(args))
    parsed_args = parser.parse_args(args)
    ic(parsed_args)
    ic(type(parsed_args))
    return parsed_args


if __name__ == "__main__":
    # SOURCE: https://stackoverflow.com/questions/2831597/processing-command-line-arguments-in-prefix-notation-in-python
    cli_args = main(sys.argv[1:])
    session = httpx.AsyncClient()
    start_time = time.time()
    # loop = asyncio.get_event_loop()
    # NOTE: https://github.com/pytest-dev/pytest-asyncio/pull/214/files#diff-cc48ab986692b5999611086a9c031ed6d88fd37496e706865aaefedb3acb9fe9
    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop_policy().get_event_loop()
    loop.run_until_complete(go_partial(loop, cli_args))
    duration = time.time() - start_time
    print(f"Computed in {duration} seconds")


# SOURCE: https://realpython.com/async-io-python/
# async def main(nprod: int, ncon: int):
#     q = asyncio.Queue()
#     producers = [asyncio.create_task(produce(n, q)) for n in range(nprod)]
#     consumers = [asyncio.create_task(consume(n, q)) for n in range(ncon)]
#     await asyncio.gather(*producers)
#     await q.join()  # Implicitly awaits consumers, too
#     for c in consumers:
#         c.cancel()
# if __name__ == "__main__":
#     import argparse
#     random.seed(444)
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-p", "--nprod", type=int, default=5)
#     parser.add_argument("-c", "--ncon", type=int, default=10)
#     ns = parser.parse_args()
#     start = time.perf_counter()
#     asyncio.run(main(**ns.__dict__))
#     elapsed = time.perf_counter() - start
#     print(f"Program completed in {elapsed:0.5f} seconds.")
