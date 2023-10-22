import asyncio
from asyncio import Semaphore
from asyncio.subprocess import Process
import os
import pathlib
import random
import string
import time
from typing import Dict, Optional

from icecream import ic
import uritools

PREPARE_FOR_IG_SMALL = """
    time ffmpeg -y \
    -hide_banner -loglevel warning \
    -i '{full_path_input_file}' \
    -c:v h264_videotoolbox \
    -bufsize 5200K \
    -b:v 5200K \
    -maxrate 5200K \
    -level 42 \
    -bf 2 \
    -g 63 \
    -refs 4 \
    -threads 16 \
    -preset:v fast \
    -vf "scale=1080:1080:force_original_aspect_ratio=decrease,pad=width=1080:height=1080:x=-1:y=-1:color=0x16202A" \
    -c:a aac \
    -ar 44100 \
    -ac 2 \
    '{full_path_output_file}'

    cp -av '{full_path_output_file}' '{source_dir}'
"""


async def run_coroutine_subprocess(
    cmd: Optional[str],
    # uri: str,
    sem: Semaphore,
    working_dir: str = f"{pathlib.Path('./').absolute()}",
):
    await asyncio.sleep(0.05)

    env = {}
    env |= os.environ

    # dl_uri = uritools.urisplit(uri)

    # full_path_input_file = "/Users/malcolm/dev/bossjones/fastapi_pytorch_postgresql_sandbox/fixtures/NoEmmeG-1705218065640694069-20230922_095042-vid1.mp4"
    # full_path_output_file = "/Users/malcolm/dev/bossjones/fastapi_pytorch_postgresql_sandbox/fixtures/NoEmmeG-1705218065640694069-20230922_095042-vid1_smaller.mp4"

    result = "0"
    cmd = f"{cmd}"

    async with sem:
        process: Process = await asyncio.create_subprocess_shell(
            cmd,
            env=env,
            cwd=working_dir,
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        # ic(stdout.decode("utf-8").strip())
        # ic(stderr.decode("utf-8").strip())
        result = stdout.decode("utf-8").strip()
        return result


async def encrypt(sem: Semaphore, text: str) -> bytes:
    program = [
        "gpg",
        "-c",
        "--batch",
        "--passphrase",
        "3ncryptm3",
        "--cipher-algo",
        "TWOFISH",
    ]

    async with sem:
        process: Process = await asyncio.create_subprocess_exec(
            *program, stdout=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(text.encode())
        return stdout


# async def main():
#     text_list = [
#         "".join(random.choice(string.ascii_letters) for _ in range(1000))
#         for _ in range(1000)
#     ]
#     semaphore = Semaphore(os.cpu_count())
#     s = time.time()
#     tasks = [asyncio.create_task(encrypt(semaphore, text)) for text in text_list]
#     encrypted_text = await asyncio.gather(*tasks)
#     e = time.time()

#     print(f"Total time: {e - s}")


# asyncio.run(main())
