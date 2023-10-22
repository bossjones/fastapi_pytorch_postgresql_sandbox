"""web.api.videocrop.views"""
# sourcery skip: avoid-global-variables
# pylint: disable=no-name-in-module
from __future__ import annotations

import asyncio
from asyncio import Semaphore
from asyncio.subprocess import Process
import concurrent.futures
import functools
from io import BytesIO
import os
import pathlib
import pickle
import random
import string
import sys
import tempfile
import time
import traceback
from typing import Union
import uuid

from PIL import Image
from aio_pika import Channel, DeliveryMode, ExchangeType, Message
from aio_pika.pool import Pool
import aiofiles
import bpdb
from codetiming import Timer
from fastapi import APIRouter, Depends, UploadFile, status
from fastapi.responses import JSONResponse
from icecream import ic
from redis.asyncio import ConnectionPool, Redis
import rich

from fastapi_pytorch_postgresql_sandbox.services.rabbit.dependencies import (
    get_rmq_channel_pool,
)
from fastapi_pytorch_postgresql_sandbox.services.redis.dependency import get_redis_pool
from fastapi_pytorch_postgresql_sandbox.settings import settings
from fastapi_pytorch_postgresql_sandbox.utils.imgops import (
    handle_save_attachment_locally,
)
from fastapi_pytorch_postgresql_sandbox.utils.mlops import (
    convert_pil_image_to_rgb_channels,
)
from fastapi_pytorch_postgresql_sandbox.utils.shell import run_coroutine_subprocess
from fastapi_pytorch_postgresql_sandbox.web.api.screennet.schema import (
    PendingClassificationDTO,
    RedisPredictionValueDTO,
)

# NOTE: Inspired by https://www.auroria.io/running-pytorch-models-for-inference-using-fastapi-rabbitmq-redis-docker/


def aio_convert_pil_image_to_rgb_channels(images_filepath: str) -> Image:
    """_summary_

    Args:
        images_filepath (str): _description_

    Returns:
        Image: _description_
    """
    rich.print(f"converting pil image to rgb channels via async ...{images_filepath}")
    image_data: Image = convert_pil_image_to_rgb_channels(images_filepath)
    return image_data


router = APIRouter()


@router.post("/crop", response_model=PendingClassificationDTO, status_code=202)
async def crop(
    # message: RMQMessageDTO,
    file: UploadFile,
    pool: Pool[Channel] = Depends(get_rmq_channel_pool),
) -> PendingClassificationDTO:
    """
    Posts a message in a rabbitMQ's exchange.

    :param message: message to publish to rabbitmq.
    :param pool: rabbitmq channel pool
    """

    request_object_content = await file.read()

    # rich.inspect(file, all=True)

    hash_prefix = uuid.uuid4().hex

    video_container = {
        "filename": f"{file.filename}",
        "content_type": f"{file.content_type}",
        "prefix": hash_prefix,
        "data": request_object_content,
        "upload_file_obj": file,
    }

    # contents = await file.read()
    # pil_image = Image.open(BytesIO(image_payload_bytes))  # orig
    # pil_image: Image = Image.open(file.file)  # orig
    # pil_image: Image = Image.open(contents)  # orig
    # SOURCE: https://github.com/tiangolo/fastapi/discussions/4308
    # pil_image: Image = Image.open(BytesIO(request_object_content))

    # rich.inspect(file, all=True)
    video_filepaths = []

    semaphore = Semaphore(os.cpu_count())

    # HACK: write file to disk first
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("created temporary directory", tmpdirname)
        with Timer(text="\nTotal elapsed time: {:.1f}"):
            local_attachment_path = await handle_save_attachment_locally(
                video_container,
                tmpdirname,
            )
            async with aiofiles.open(local_attachment_path, "wb") as out_file:
                await out_file.write(request_object_content)
            video_filepaths.append(local_attachment_path)

            for _, video_fpaths in enumerate(video_filepaths):
                try:
                    # if hasattr(pil_image, "filename"):
                    #     # FIXME: # we need this
                    # print(f"{file.file}")
                    await asyncio.sleep(1)
                    print(f"video_fpaths: {video_fpaths}")
                    ic(video_fpaths)
                    # s = time.time()
                    # tasks = [asyncio.create_task(run_coroutine_subprocess(cmd, uri, semaphore, text)) for text in text_list]
                    # encrypted_text = await asyncio.gather(*tasks)
                    # e = time.time()
                    # print(f"Total time: {e - s}")
                    # # nuke the originals
                    # convert_func = functools.partial(
                    #     convert_pil_image_to_rgb_channels,
                    #     video_fpaths,
                    # )

                    # # 2. Run in a custom thread pool:
                    # with concurrent.futures.ThreadPoolExecutor() as cp:
                    #     loop = asyncio.get_running_loop()
                    #     video_data = await loop.run_in_executor(cp, convert_func)
                    #     # rich.print(f"count: {count} - Unlink", unlink_result)
                    # await asyncio.sleep(1)

                except Exception as ex:
                    print(f"{ex}")
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    print(f"Error Class: {ex.__class__}")
                    output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
                    print(output)
                    print(f"exc_type: {exc_type}")
                    print(f"exc_value: {exc_value}")
                    traceback.print_tb(exc_traceback)
                    bpdb.pm()

                inference_id = str(uuid.uuid4())

                # TODO: need to re-enable this once we upload the video, modify it, then store it in rabbitmq
                # async with pool.acquire() as conn:
                #     exchange = await conn.declare_exchange(
                #         # message.exchange_name,
                #         "videocrop",
                #         # SOURCE: https://aio-pika.readthedocs.io/en/latest/rabbitmq-tutorial/4-routing.html#multiple-bindings
                #         type=ExchangeType.DIRECT,
                #         # name="",  # default exchange
                #         auto_delete=True,
                #     )
                #     # Declaring queue
                #     queue = await conn.declare_queue(
                #         settings.worker_queue_name,
                #         # NOTE: When RabbitMQ quits or crashes it will forget the queues and messages unless you tell it not to. Two things are required to make sure that messages aren't lost: we need to mark both the queue and messages as durable.
                #         # NOTE: First, we need to make sure that RabbitMQ will never lose our queue. In order to do so, we need to declare it as durable:
                #         durable=True,
                #     )

                #     # NOTE: https://aio-pika.readthedocs.io/en/latest/rabbitmq-tutorial/4-routing.html#multiple-bindings
                #     # A binding is a relationship between an exchange and a queue. This can be simply read as: the queue is interested in messages from this exchange.
                #     # Binding the queue to the exchange
                #     await queue.bind(exchange, routing_key="crop_worker")

                #     await exchange.publish(
                #         message=Message(
                #             body=pickle.dumps(video_data),
                #             headers={"inference_id": inference_id},
                #             delivery_mode=DeliveryMode.PERSISTENT,
                #         ),
                #         routing_key="crop_worker",
                #     )

    return PendingClassificationDTO(inference_id=inference_id)


@router.get(
    "/result/{inference_id}",
    status_code=200,
    response_model=Union[RedisPredictionValueDTO, PendingClassificationDTO],
)
async def get_crop_value(
    inference_id: str,
    redis_pool: ConnectionPool = Depends(get_redis_pool),
) -> Union[RedisPredictionValueDTO, PendingClassificationDTO, JSONResponse]:
    """
    Get value from redis.

    :param key: redis key, to get data from.
    :param redis_pool: redis connection pool.
    :returns: information from redis.
    """

    async with Redis(connection_pool=redis_pool) as redis:
        exists = await redis.exists(inference_id)
        if not exists:
            pending_classification_dto = {"inference_id": inference_id}

            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content=pending_classification_dto,
            )

        redis_value = await redis.hgetall(inference_id)
    return RedisPredictionValueDTO(data=redis_value)
