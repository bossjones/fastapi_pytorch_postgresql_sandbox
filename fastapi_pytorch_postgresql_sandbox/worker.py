"""worker"""
# sourcery skip: avoid-global-variables, snake-case-arguments, require-parameter-annotation
# pylint: disable=no-name-in-module
# SOURCE: https://aio-pika.readthedocs.io/en/latest/patterns.html#master-worker
import asyncio
import pickle

# from bridge import rabbitmq_client, redis_client
import aio_pika
from aio_pika.abc import (
    AbstractChannel,
    AbstractIncomingMessage,
    AbstractRobustConnection,
)
from aio_pika.patterns import NackMessage, RejectMessage
from aio_pika.pool import Pool
from redis.asyncio import ConnectionPool, Redis
import rich

from fastapi_pytorch_postgresql_sandbox.deeplearning.architecture.screennet.ml_model import (
    ImageClassifier,
)
from fastapi_pytorch_postgresql_sandbox.logging import configure_logging
from fastapi_pytorch_postgresql_sandbox.settings import settings

configure_logging()


def init_worker_rabbit() -> (
    tuple[
        Pool[AbstractRobustConnection],
        Pool[AbstractChannel],
    ]
):  # pragma: no cover
    """
    Initialize rabbitmq pools.

    :param app: current FastAPI application.
    """

    async def get_connection() -> AbstractRobustConnection:  # noqa: WPS430
        """
        Creates connection to RabbitMQ using url from settings.

        :return: async connection to RabbitMQ.
        """
        return await aio_pika.connect_robust(str(settings.rabbit_url))

    # This pool is used to open connections.
    connection_pool: Pool[AbstractRobustConnection] = Pool(
        get_connection,
        max_size=settings.rabbit_pool_size,
    )

    async def get_channel() -> AbstractChannel:  # noqa: WPS430
        """
        Open channel on connection.

        Channels are used to actually communicate with rabbitmq.

        :return: connected channel.
        """
        async with connection_pool.acquire() as connection:
            return await connection.channel()

    # This pool is used to open channels.
    channel_pool: Pool[AbstractChannel] = Pool(
        get_channel,
        max_size=settings.rabbit_channel_pool_size,
    )

    return connection_pool, channel_pool


async def shutdown_worker_rabbit(
    connection_pool: Pool[AbstractRobustConnection],
    channel_pool: Pool[AbstractChannel],
) -> None:  # pragma: no cover
    """Close all connection and pools.

    Args:
        connection_pool (Pool[AbstractRobustConnection]): _description_
        channel_pool (Pool[AbstractChannel]): _description_
    """
    await connection_pool.close()
    await channel_pool.close()


def init_worker_redis() -> ConnectionPool:  # pragma: no cover
    """
    Creates connection pool for redis.
    """
    redis_pool: ConnectionPool = ConnectionPool.from_url(
        str(settings.redis_url),
    )

    return redis_pool


async def shutdown_worker_redis(redis_pool: ConnectionPool) -> None:  # pragma: no cover
    """
    Closes redis connection pool.

    :param redis_pool: redis ConnectionPool.
    """
    await redis_pool.disconnect()


async def worker(*, task_id: int) -> None:
    """_summary_

    Args:
        task_id (int): _description_

    Raises:
        RejectMessage: _description_
        NackMessage: _description_
    """
    # If you want to reject message or send
    # nack you might raise special exception

    if task_id % 2 == 0:
        raise RejectMessage(requeue=False)

    if task_id % 2 == 1:
        raise NackMessage(requeue=False)

    print(task_id)


async def on_message(message: AbstractIncomingMessage) -> None:
    """_summary_

    Args:
        message (AbstractIncomingMessage): _description_
    """
    # NOTE: Example message data
    # IncomingMessage:{'app_id': None,
    #  'body_size': 10963333,
    #  'cluster_id': '',
    #  'consumer_tag': 'ctag1.bdd640fb06674ad19c80317fa3b1799d',
    #  'content_encoding': None,
    #  'content_type': None,
    #  'correlation_id': None,
    #  'delivery_mode': <DeliveryMode.PERSISTENT: 2>,
    #  'delivery_tag': 1,
    #  'exchange': 'screenet',
    #  'expiration': None,
    #  'headers': {'inference_id': '6f226399-bf6b-4aa2-8a73-586f6627eaff',
    #              'traceparent': '00-bdd640fb06671ad11c80317fa3b1799d-371ecd7b27cd8130-01'},
    #  'message_id': '1a2a73ed562b4f79837459eef50bea63',
    #  'priority': 0,
    #  'redelivered': False,
    #  'reply_to': None,
    #  'routing_key': 'classify_worker',
    #  'timestamp': None,
    #  'type': 'None',
    #  'user_id': None}
    async with message.process():
        # import bpdb

        # bpdb.set_trace()
        image_data = pickle.loads(message.body)

        # HACK: For some reason the image isn't being converted properly in the view route, lets catch it here just in case and convert to what we expect to use
        if image_data.mode == "RGBA":
            image_data = image_data.convert("RGB")

        result = net_api.infer(image_data)
        rich.print(f"Result: {result}")
        # Looks like this # Result: [{'pred_prob': 0.439, 'pred_class': 'twitter', 'time_for_pred': 4.2298}]

        # SOURCE: https://www.auroria.io/running-pytorch-models-for-inference-using-fastapi-rabbitmq-redis-docker/
        # Storing prediction in redis database
        # FIXME: set the values using redis now
        if result is not None:
            async with Redis(connection_pool=redis_pool) as redis:
                await redis.hset(
                    name=message.headers["inference_id"], mapping=result[0],
                )
                # print("")


async def main() -> None:
    """_summary_"""
    rabbit_connection_pool, rabbit_channel_pool = init_worker_rabbit()

    # rich.print(rabbit_connection_pool)
    # rich.print(rabbit_channel_pool)

    async with rabbit_connection_pool.acquire() as connection:
        async with rabbit_channel_pool.acquire() as channel:
            await channel.set_qos(prefetch_count=1)

            # -------------------------------------------------------
            # Message durability
            # We have learned how to make sure that even if the consumer dies, the task isn't lost. But our tasks will still be lost if RabbitMQ server stops.
            # When RabbitMQ quits or crashes it will forget the queues and messages unless you tell it not to. Two things are required to make sure that messages aren't lost: we need to mark both the queue and messages as durable.
            # First, we need to make sure that RabbitMQ will never lose our queue. In order to do so, we need to declare it as durable:
            # Initializing Master with channel
            # -------------------------------------------------------
            # SOURCE: https://aio-pika.readthedocs.io/en/latest/rabbitmq-tutorial/2-work-queues.html

            # Declaring queue
            queue = await channel.declare_queue(
                # "screennet_inference_queue",
                settings.worker_queue_name,
                durable=True,
            )

            # https://aio-pika.readthedocs.io/en/latest/rabbitmq-tutorial/2-work-queues.html

            # Start listening the queue with name 'task_queue'
            await queue.consume(on_message)

            # master = Master(channel)
            # await master.create_worker("fastapiworker", worker, auto_delete=True)

            print(" [*] Waiting for messages. To exit press CTRL+C")
            try:
                await asyncio.Future()
            finally:
                await connection.close()


if __name__ == "__main__":
    redis_pool = init_worker_redis()
    net_api = ImageClassifier()
    net_api.load_model()
    asyncio.run(main())
