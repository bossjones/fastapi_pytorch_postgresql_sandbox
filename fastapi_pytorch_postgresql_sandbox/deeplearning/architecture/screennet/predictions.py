""" predictions """
from __future__ import annotations

import argparse

# SOURCE: https://github.com/rasbt/deeplearning-models/blob/35aba5dc03c43bc29af5304ac248fc956e1361bf/pytorch_ipynb/helper_evaluate.py
import pathlib
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, Union
from urllib.parse import urlparse

import pandas as pd
import requests
import torch
import torch.optim
import torch.profiler
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from fastai.data.transforms import get_image_files
from icecream import ic
from rich import print
from tqdm.auto import tqdm

from fastapi_pytorch_postgresql_sandbox.utils.mlops import (
    console_print_table,
    convert_pil_image_to_rgb_channels,
    from_pil_image_to_plt_display,
    write_predict_results_to_csv,
)


# SOURCE: https://www.learnpytorch.io/09_pytorch_model_deployment/
# 1. Create a function to return a list of dictionaries with sample, truth label, prediction, prediction probability and prediction time
def pred_and_store(
    paths: List[pathlib.Path],
    model: torch.nn.Module,
    transform: torchvision.transforms,
    class_names: List[str],
    device: Union[str, torch.device] = "",
) -> List[Dict]:
    """_summary_

    Args:
        paths (List[pathlib.Path]): _description_
        model (torch.nn.Module): _description_
        transform (torchvision.transforms): _description_
        class_names (List[str]): _description_
        device (torch.device, optional): _description_. Defaults to "".

    Returns:
        List[Dict]: _description_
    """

    ic(paths)
    ic(model.name)
    ic(transform)
    ic(class_names)
    ic(device)
    # 2. Create an empty list to store prediction dictionaires
    pred_list = []

    # 3. Loop through target paths
    for path in tqdm(paths):
        class_name = path.parent.stem
        pred_d = {"image_path": path, "class_name": class_name}
        # 6. Start the prediction timer
        start_time = timer()

        # 7. Open image path
        # img = Image.open(path)
        img = convert_pil_image_to_rgb_channels(f"{paths[0]}")

        # 8. Transform the image, add batch dimension and put image on target device
        # transformed_image = transform(img).unsqueeze(dim=0).to(device)
        transformed_image = transform(img).unsqueeze(dim=0)

        # 9. Prepare model for inference by sending it to target device and turning on eval() mode
        model.to(device)
        model.eval()

        # 10. Get prediction probability, predicition label and prediction class
        with torch.inference_mode():
            pred_logit = model(
                transformed_image.to(device),
            )  # perform inference on target sample
            pred_prob = torch.softmax(
                pred_logit,
                dim=1,
            )  # turn logits into prediction probabilities
            pred_label = torch.argmax(
                pred_prob,
                dim=1,
            )  # turn prediction probabilities into prediction label
            pred_class = class_names[
                pred_label.cpu()
            ]  # hardcode prediction class to be on CPU

            # 11. Make sure things in the dictionary are on CPU (required for inspecting predictions later on)
            pred_d["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
            pred_d["pred_class"] = pred_class

            # 12. End the timer and calculate time per pred
            end_time = timer()
            pred_d["time_for_pred"] = round(end_time - start_time, 4)

        # 13. Does the pred match the true label?
        pred_d["correct"] = class_name == pred_class

        # 14. Add the dictionary to the list of preds
        pred_list.append(pred_d)

    # 15. Return list of prediction dictionaries
    return pred_list


def predict_from_dir(
    path_to_image_from_cli: str,
    model: torch.nn.Module,
    transforms: torchvision.transforms,
    class_names: List[str],
    device: Union[str, torch.device],
    args: argparse.Namespace,
) -> None:
    """wrapper function to perform predictions on individual files

    Args:
        path_to_image_from_cli (str): eg.  "/Users/malcolm/Downloads/2020-11-25_10-47-32_867.jpeg
        model (torch.nn.Module): _description_
        transform (torchvision.transforms): _description_
        class_names (List[str]): _description_
        device (torch.device): _description_
        args (argparse.Namespace): _description_
    """
    ic(f"Predict | directory {path_to_image_from_cli} ...")
    image_folder_api = get_image_files(path_to_image_from_cli)
    ic(image_folder_api)

    paths = image_folder_api

    for paths_item in paths:
        predict_from_file(
            paths_item,
            model,
            transforms,
            class_names,
            device,
            args,
        )


def predict_from_file(
    path_to_image_from_cli: str,
    model: torch.nn.Module,
    transforms: torchvision.transforms,
    class_names: List[str],
    device: Union[str, torch.device],
    args: argparse.Namespace,
) -> None:
    """wrapper function to perform predictions on individual files

    Args:
        path_to_image_from_cli (str): eg.  "/Users/malcolm/Downloads/2020-11-25_10-47-32_867.jpeg
        model (torch.nn.Module): _description_
        transform (torchvision.transforms): _description_
        class_names (List[str]): _description_
        device (torch.device): _description_
        args (argparse.Namespace): _description_
    """
    ic(f"Predict | individual file {path_to_image_from_cli} ...")
    image_path_api = pathlib.Path(path_to_image_from_cli).resolve()
    ic(image_path_api)

    paths = [image_path_api]
    img = convert_pil_image_to_rgb_channels(f"{paths[0]}")

    pred_ds = pred_and_store(paths, model, transforms, class_names, device)

    if args.to_disk and args.results != "":
        write_predict_results_to_csv(pred_ds, args)

    image_class = pred_ds[0]["pred_class"]
    image_pred_prob = pred_ds[0]["pred_prob"]
    image_time_for_pred = pred_ds[0]["time_for_pred"]

    # 5. Print metadata
    print(f"Random image path: {paths[0]}")
    print(f"Image class: {image_class}")
    print(f"Image pred prob: {image_pred_prob}")
    print(f"Image pred time: {image_time_for_pred}")
    print(f"Image height: {img.height}")
    print(f"Image width: {img.width}")

    # print prediction info to rich table
    pred_df = pd.DataFrame(pred_ds)
    console_print_table(pred_df)

    plot_fname = (
        f"results/prediction-{model.name}-{image_path_api.stem}{image_path_api.suffix}"
    )

    from_pil_image_to_plt_display(
        img,
        pred_ds,
        to_disk=args.to_disk,
        interactive=args.interactive,
        fname=plot_fname,
    )


def download_and_predict(
    url: str,
    model: torch.nn.Module,
    data_path: pathlib.PosixPath,
    class_names: List[str],
    device: Union[str, torch.device, None] = None,
) -> None:
    """_summary_

    Args:
        url (str): _description_
        model (torch.nn.Module): _description_
        data_path (pathlib.PosixPath): _description_
        class_names (List[str]): _description_
        device (torch.device, optional): _description_. Defaults to None.
    """
    # Download custom image
    # urlparse(url).path
    fname = Path(urlparse(url).path).name

    # Setup custom image path
    custom_image_path = data_path / fname

    print(f"fname: {custom_image_path}")

    # Download the image if it doesn't already exist
    if not custom_image_path.is_file():
        with open(custom_image_path, "wb") as f:
            # When downloading from GitHub, need to use the "raw" file link
            request = requests.get(url)
            print(f"Downloading {custom_image_path}...")
            f.write(request.content)
    else:
        print(f"{custom_image_path} already exists, skipping download.")


# TODO: Re-enable this
# def execute_prediction(
#     self,
#     path_to_image: str,
#     model: torch.nn.Module,
#     class_names: List[str],
#     device: torch.device = None,
# ) -> None:
#     """_summary_"""
#     # if args.download_and_predict:
#     #     print(" Running download and predict command ...")
#     #     download_and_predict(
#     #         args.download_and_predict,
#     #         model,
#     #         Path(args.data),
#     #         class_names=class_names,
#     #         device=device,
#     #     )
#     #     return

#     # if args.predict:
#     print(" Running predict command ...")
#     # 3. Get image class from path name (the image class is the name of the directory where the image is stored)
#     path_to_image_from_cli = fix_path(path_to_image)

#     if is_file(path_to_image_from_cli):
#         predict_from_file(
#             path_to_image_from_cli,
#             model,
#             auto_transforms,
#             self.class_names,
#             device,
#             args,
#         )

#     if is_directory(path_to_image_from_cli):
#         predict_from_dir(
#             path_to_image_from_cli,
#             model,
#             auto_transforms,
#             self.class_names,
#             device,
#             args,
#         )
#     return
