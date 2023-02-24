""" file_functions """
from __future__ import annotations

import os
import os.path
import pathlib
import shutil
from pathlib import Path
from typing import Any

from rich import print


# SOURCE: https://github.com/tgbugs/pyontutils/blob/05dc32b092b015233f4a6cefa6c157577d029a40/ilxutils/tools.py
def is_file(path: str) -> bool:
    """Check if path contains a file

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    return bool(pathlib.Path(path).is_file())


def is_directory(path: str) -> bool:
    """Check if path contains a dir

    Args:
        path (str): _description_

    Returns:
        _type_: _description_
    """
    return bool(pathlib.Path(path).is_dir())


def tilda(obj: Any):
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


def fix_path(path: str):
    """Automatically convert path to fully qualifies file uri.

    Args:
        path (_type_): _description_
    """

    def __fix_path(path):
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


def walk_through_dir(dir_path: str):
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
