import os
import pathlib
import uuid

from aio_pika.abc import AbstractQueue
from fastapi import FastAPI
from httpx import AsyncClient
import pytest

HERE = os.path.dirname(__file__)

# @pytest.mark.xfail(
#     reason="wip",
# )
@pytest.mark.anyio
async def test_message_publishing_classify(
    fastapi_app: FastAPI,
    client: AsyncClient,
    test_queue: AbstractQueue,
    test_exchange_name: str,
    test_routing_key: str,
) -> None:
    """
    Tests that message is published correctly.

    It sends message to rabbitmq and reads it
    from binded queue.
    """
    message_text = uuid.uuid4().hex
    url = fastapi_app.url_path_for("classify")
    # NOTE: https://aio-pika.readthedocs.io/en/latest/rabbitmq-tutorial/4-routing.html#multiple-bindings
    path = pathlib.Path(
        f"{HERE}/fixtures/test1.jpg",
    )
    with path.open("rb") as file:
        # import bpdb

        # bpdb.set_trace()
        await client.post(
            url,
            # json={
            #     "exchange_name": test_exchange_name,  # default exchange
            #     "routing_key": test_routing_key,
            #     # NOTE: These are the production values below
            #     # "exchange_name": "", # default exchange
            #     # "routing_key": "screennet_inference_queue",
            #     "queue_name": test_queue.name,
            #     "message": message_text,
            # },
            # files={"file": file},
            files={"file": ("filename", file, "image/jpeg")},
            headers={
                "accept": "application/json",
                "Content-Type": "multipart/form-data",
            },
        )

        # print("hello")
    # message = await test_queue.get(timeout=1)
    # assert message is not None
    # await message.ack()
    # assert message.body.decode("utf-8") == message_text


# @pytest.mark.anyio
# async def test_message_wrong_exchange(
#     fastapi_app: FastAPI,
#     client: AsyncClient,
#     test_queue: AbstractQueue,
#     test_exchange_name: str,
#     test_routing_key: str,
#     test_rmq_pool: Pool[Channel],
# ) -> None:
#     """
#     Tests that message can be published in undeclared exchange.

#     It sends message to random queue,
#     tries to get message from binded queue
#     and checks that new exchange were created.
#     """
#     random_exchange = uuid.uuid4().hex
#     assert random_exchange != test_exchange_name
#     message_text = uuid.uuid4().hex
#     url = fastapi_app.url_path_for("send_rabbit_message")
#     await client.post(
#         url,
#         json={
#             "exchange_name": random_exchange,
#             "routing_key": test_routing_key,
#             "message": message_text,
#         },
#     )
#     with pytest.raises(QueueEmpty):
#         await test_queue.get(timeout=1)

#     async with test_rmq_pool.acquire() as conn:
#         exchange = await conn.get_exchange(random_exchange, ensure=True)
#         await exchange.delete(if_unused=False)
