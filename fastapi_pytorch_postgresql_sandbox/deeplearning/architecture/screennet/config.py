""" devices """
from __future__ import annotations

import os
import os.path
import pathlib

import better_exceptions
import bpdb
import pandas as pd

# import devices  # pylint: disable=import-error
import rich

# ---------------------------------------------------------------------------
import torch
import torchvision
from icecream import ic
from rich import box, inspect, print
from rich.console import Console
from rich.table import Table
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Import rich and whatever else we need
# %load_ext rich
# %matplotlib inline




# better_exceptions.hook()

# console: Console = Console()
# ---------------------------------------------------------------------------


assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
assert (
    int(torchvision.__version__.split(".")[1]) >= 13
), "torchvision version should be 0.13+"
# print(f"torch version: {torch.__version__}")
# print(f"torchvision version: {torchvision.__version__}")
# ---------------------------------------------------------------------------

# Continue with regular imports
import matplotlib.pyplot as plt
import mlxtend
import torch
import torchmetrics
import torchvision
from torch import nn
from torchinfo import summary
from torchvision import transforms

# breakpoint()
# from going_modular import data_setup, engine, utils  # pylint: disable=no-name-in-module

# Try to get torchinfo, install it if it doesn't work


# print(f"mlxtend version: {mlxtend.__version__}")
assert (
    int(mlxtend.__version__.split(".")[1]) >= 19
), "mlxtend verison should be 0.19.0 or higher"

import os
from itertools import product
from pathlib import Path

import fastai
import matplotlib
import numpy as np
import numpy.typing as npt
import PIL
import requests

# SOURCE: https://github.com/rasbt/deeplearning-models/blob/35aba5dc03c43bc29af5304ac248fc956e1361bf/pytorch_ipynb/helper_evaluate.py
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim

# # Import accuracy metric
# from helper_functions import (  # Note: could also use torchmetrics.Accuracy()
#     accuracy_fn,
#     plot_loss_curves,
# )
import torch.profiler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as pytorch_transforms_functional
from fastai.data.transforms import get_image_files
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import ConfusionMatrix
from watermark import watermark

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
# print(model_names)

shared_datasets_path_api = pathlib.Path(os.path.expanduser("~/Downloads/datasets"))
shared_datasets_path = os.path.abspath(str(shared_datasets_path_api))
# print(f"shared_datasets_path - {shared_datasets_path}")
DEFAULT_DATASET_DIR = Path(f"{shared_datasets_path}")

best_acc1 = 0

PATH_TO_BEST_MODEL_API = pathlib.Path(
    os.path.expanduser("~/ScreenCropNetV1_378_epochs.pth"),
)
PATH_TO_BEST_MODEL = os.path.abspath(str(PATH_TO_BEST_MODEL_API))
