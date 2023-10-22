""" file_functions """
# pylint: disable=consider-using-sys-exit
from __future__ import annotations

import os
import os.path
import pathlib
from pathlib import Path
import shutil
from typing import Any, Generator, List

import aiofiles
import aiofiles.os
from fastai.data.transforms import get_files, get_image_files

# from fastapi.responses import StreamingResponse
from rich import print

# PathLike = TypeVar("PathLike", str, pathlib.Path, None)
PathLike = str | pathlib.Path | None


def go_get_image_files(path_to_image_from_cli: str) -> List[PathLike]:
    """Leverage fastai's get_image_files function.

    Args:
        path_to_image_from_cli (str): str of directory

    Returns:
        List[str]: List of image files
    """
    return get_image_files(path_to_image_from_cli, recurse=True)


def get_video_files(path, recurse=True, folders=None):
    "Get text files in `path` recursively, only in `folders`, if specified."
    return get_files(path, extensions=[".mp4"], recurse=recurse, folders=folders)


def go_get_video_files(path_to_videos_from_cli: str) -> List[PathLike]:
    """Leverage fastai's get_video_files function.

    Args:
        path_to_videos_from_cli (str): str of directory

    Returns:
        List[str]: List of image files
    """
    return get_video_files(path_to_videos_from_cli, recurse=True)


# SOURCE: https://github.com/tgbugs/pyontutils/blob/05dc32b092b015233f4a6cefa6c157577d029a40/ilxutils/tools.py
def is_file(path: str) -> bool:
    """Check if path contains a file

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    return pathlib.Path(path).is_file()


def is_directory(path: str) -> bool:
    """Check if path contains a dir

    Args:
        path (str): _description_

    Returns:
        _type_: _description_
    """
    return pathlib.Path(path).is_dir()


def tilda(obj: Any) -> List[str | Any] | str | Any:
    """wrapper for linux ~/ shell notation

    Args:
        obj (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(obj, list):
        return [
            str(pathlib.Path(o).expanduser()) if isinstance(o, str) else o for o in obj
        ]
    elif isinstance(obj, str):
        return str(pathlib.Path(obj).expanduser())
    else:
        return obj


def fix_path(path: str) -> Any:
    """Automatically convert path to fully qualifies file uri.

    Args:
        path (_type_): _description_
    """

    # Either all return statements in a function should return an expression, or none of them should.
    def __fix_path(path):  # pylint: disable=inconsistent-return-statements
        if not isinstance(path, str):
            return path
        elif "~" == path[0]:
            tilda_fixed_path = tilda(path)
            if is_file(tilda_fixed_path):
                return tilda_fixed_path
            else:
                exit(path, ": does not exit.")
        elif is_file(pathlib.Path.home() / path):
            return str(pathlib.Path().home() / path)
        elif is_directory(pathlib.Path.home() / path):
            return str(pathlib.Path().home() / path)
        else:
            return path

    if isinstance(path, str):
        return __fix_path(path)
    elif isinstance(path, list):
        return [__fix_path(p) for p in path]
    else:
        return path


def walk_through_dir(dir_path: str) -> List[str]:
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str or pathlib.Path): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.",
        )


def clean_dir_images(image_path: str) -> None:
    """_summary_

    Args:
        image_path (_type_): _description_
    """
    for f in Path(image_path).glob("*.jpg"):
        try:
            f.unlink()
        except OSError as e:
            print(f"Error: {f} : {e.strerror}")


def clean_dirs_in_dir(image_path: str) -> None:
    """_summary_

    Args:
        image_path (_type_): _description_
    """
    try:
        shutil.rmtree(image_path)
    except OSError as e:
        print(f"Error: {image_path} : {e.strerror}")


# SOURCE: https://github.com/objective-core/objective_backend/blob/f721a950e001b8a4044b93c984f8c7a9353a36e3/thumbnailer/main.py
async def aio_is_path_exists(path: str) -> bool:
    """_summary_

    Args:
        path (str): _description_

    Returns:
        bool: _description_
    """
    try:
        await aiofiles.os.stat(path)
    except (OSError, ValueError):
        return False
    return True


# SOURCE: https://github.com/objective-core/objective_backend/blob/f721a950e001b8a4044b93c984f8c7a9353a36e3/thumbnailer/main.py
# https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
# If your generator will only yield values, set the SendType and ReturnType to None
def iterfile(file_path: str) -> Generator[Any, None, None]:
    """If you have a file-like object (e.g. the object returned by open()), you can create a generator function to iterate over that file-like object.

    That way, you don't have to read it all first in memory, and you can pass that generator function to the StreamingResponse, and return it.

    This includes many libraries to interact with cloud storage, video processing, and others.

    Args:
        file_path (_type_): _description_

    Yields:
        _type_: _description_
    """
    with open(file_path, mode="rb") as file_like:
        yield from file_like
