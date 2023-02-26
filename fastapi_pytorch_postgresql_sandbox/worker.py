"""worker"""
# sourcery skip: avoid-global-variables, snake-case-arguments, require-parameter-annotation
# pylint: disable=no-name-in-module
# SOURCE: https://aio-pika.readthedocs.io/en/latest/patterns.html#master-worker
import asyncio

# from bridge import rabbitmq_client, redis_client
import aio_pika
from aio_pika.abc import AbstractChannel, AbstractRobustConnection
from aio_pika.patterns import Master, NackMessage, RejectMessage
from aio_pika.pool import Pool
from redis.asyncio import ConnectionPool
import rich

from fastapi_pytorch_postgresql_sandbox.settings import settings

# from pika import BasicProperties
# from inference_model import MNISTInferenceModel


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


def init_worker_redis() -> None:  # pragma: no cover
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


async def main() -> None:
    """_summary_"""
    rabbit_connection_pool, rabbit_channel_pool = init_worker_rabbit()

    rich.print(rabbit_connection_pool)
    rich.print(rabbit_channel_pool)
    # connection = await connect_robust(
    #     "amqp://guest:guest@127.0.0.1/?name=aio-pika%20worker",
    # )

    #     # # Creating channel
    #     # channel = await connection.channel()

    async with rabbit_connection_pool.acquire() as connection:
        async with rabbit_channel_pool.acquire() as channel:
            # Initializing Master with channel
            master = Master(channel)
            await master.create_worker("fastapiworker", worker, auto_delete=True)

            try:
                await asyncio.Future()
            finally:
                await connection.close()


if __name__ == "__main__":
    asyncio.run(main())