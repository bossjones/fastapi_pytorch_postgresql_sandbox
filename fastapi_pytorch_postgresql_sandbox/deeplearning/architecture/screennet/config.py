# sourcery skip: docstrings-for-classes, avoid-global-variables
""" devices """
from __future__ import annotations

import os
import os.path
import pathlib
from pathlib import Path

import torchvision.models as torchvision_models

# SOURCE: https://github.com/rasbt/deeplearning-models/blob/35aba5dc03c43bc29af5304ac248fc956e1361bf/pytorch_ipynb/helper_evaluate.py
# Continue with regular imports
# import mlxtend
# ---------------------------------------------------------------------------
# import torch
# import torch.nn.parallel
# import torch.optim

# # Import accuracy metric
# from helper_functions import (  # Note: could also use torchmetrics.Accuracy()
#     accuracy_fn,
#     plot_loss_curves,
# )
# import torch.profiler


model_names = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)
# print(model_names)

shared_datasets_path_api = pathlib.Path(os.path.expanduser("~/Downloads/datasets"))
shared_datasets_path = os.path.abspath(str(shared_datasets_path_api))
# print(f"shared_datasets_path - {shared_datasets_path}")
DEFAULT_DATASET_DIR = Path(f"{shared_datasets_path}")

best_acc1 = 0

PATH_TO_BEST_MODEL_API = pathlib.Path(
    os.path.expanduser("~/Documents/my_models/ScreenNetV1.pth"),
)
assert PATH_TO_BEST_MODEL_API.exists()

PATH_TO_BEST_MODEL = os.path.abspath(str(PATH_TO_BEST_MODEL_API))
