"""web.api.screennet.views"""
# sourcery skip: avoid-global-variables
# pylint: disable=no-name-in-module
import asyncio
from typing import Any

import rich


async def handle_save_attachment_locally(img_container: dict, dir_root: Any) -> str:
    """async function to write file to disk

    Args:
        img_container (dict): _description_
        dir_root (Any): _description_

    Returns:
        _type_: _description_
    """
    fname = f"{dir_root}/{img_container['prefix']}_{img_container['filename']}"
    rich.print(f"Saving to ... {fname}")
    await img_container["upload_file_obj"].write(img_container["data"])
    await asyncio.sleep(1)
    return fname
