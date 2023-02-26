"""web.api.screennet.views"""
# sourcery skip: avoid-global-variables
# pylint: disable=no-name-in-module
import pickle
import uuid

from PIL import Image
from aio_pika import Channel, DeliveryMode, ExchangeType, Message
from aio_pika.pool import Pool
from fastapi import APIRouter, Depends, UploadFile

from fastapi_pytorch_postgresql_sandbox.services.rabbit.dependencies import (
    get_rmq_channel_pool,
)
from fastapi_pytorch_postgresql_sandbox.utils.mlops import (
    convert_pil_image_to_rgb_channels,
)
from fastapi_pytorch_postgresql_sandbox.web.api.rabbit.schema import RMQMessageDTO
from fastapi_pytorch_postgresql_sandbox.web.api.screennet.schema import (
    PendingClassificationDTO,
)

router = APIRouter()


@router.post("/classify", response_model=PendingClassificationDTO, status_code=202)
# async def send_rabbit_message(
async def classify(
    message: RMQMessageDTO,
    file: UploadFile,
    pool: Pool[Channel] = Depends(get_rmq_channel_pool),
) -> None:
    """
    Posts a message in a rabbitMQ's exchange.

    :param message: message to publish to rabbitmq.
    :param pool: rabbitmq channel pool
    """
    # pil_image = Image.open(BytesIO(image_payload_bytes))  # orig
    pil_image: Image = Image.open(file.file)  # orig

    image_data: Image = convert_pil_image_to_rgb_channels(pil_image)

    inference_id = str(uuid.uuid4())

    async with pool.acquire() as conn:
        exchange = await conn.declare_exchange(
            message.exchange_name,
            # SOURCE: https://aio-pika.readthedocs.io/en/latest/rabbitmq-tutorial/4-routing.html#multiple-bindings
            type=ExchangeType.DIRECT,
            # name="",  # default exchange
            auto_delete=True,
        )
        # Declaring queue
        queue = await conn.declare_queue(
            message.queue_name,
            # "screennet_inference_queue",
            # properties=BasicProperties(headers={'inference_id': inference_id})
            # NOTE: When RabbitMQ quits or crashes it will forget the queues and messages unless you tell it not to. Two things are required to make sure that messages aren't lost: we need to mark both the queue and messages as durable.
            # NOTE: First, we need to make sure that RabbitMQ will never lose our queue. In order to do so, we need to declare it as durable:
            durable=True,
        )

        # NOTE: https://aio-pika.readthedocs.io/en/latest/rabbitmq-tutorial/4-routing.html#multiple-bindings
        # A binding is a relationship between an exchange and a queue. This can be simply read as: the queue is interested in messages from this exchange.
        # Binding the queue to the exchange
        await queue.bind(exchange, routing_key=message.routing_key)

        await exchange.publish(
            message=Message(
                # body=message.message.encode("utf-8"),
                body=pickle.dumps(image_data),
                # content_encoding="utf-8",
                # content_type="text/plain",
                # correlation_id: Useful to correlate RPC responses with requests.
                # correlation_id
                headers={"inference_id": inference_id},
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=message.routing_key,
        )


# @router.get("/", response_model=RedisValueDTO)
# async def get_redis_value(
#     key: str,
#     redis_pool: ConnectionPool = Depends(get_redis_pool),
# ) -> RedisValueDTO:
#     """
#     Get value from redis.

#     :param key: redis key, to get data from.
#     :param redis_pool: redis connection pool.
#     :returns: information from redis.
#     """
#     async with Redis(connection_pool=redis_pool) as redis:
#         redis_value = await redis.get(key)
#     return RedisValueDTO(
#         key=key,
#         value=redis_value,
#     )


# @router.put("/")
# async def set_redis_value(
#     redis_value: RedisValueDTO,
#     redis_pool: ConnectionPool = Depends(get_redis_pool),
# ) -> None:
#     """
#     Set value in redis.

#     :param redis_value: new value data.
#     :param redis_pool: redis connection pool.
#     """
#     if redis_value.value is not None:
#         async with Redis(connection_pool=redis_pool) as redis:
#             await redis.set(name=redis_value.key, value=redis_value.value)
