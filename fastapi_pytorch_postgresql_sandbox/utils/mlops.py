import os
import os.path

# Continue with regular imports
import matplotlib.pyplot as plt
import mlxtend
import pandas as pd

# ---------------------------------------------------------------------------
import torch
from icecream import ic
from rich import box, print
from rich.table import Table
from torchvision import transforms

assert (
    int(mlxtend.__version__.split(".")[1]) >= 19
), "mlxtend verison should be 0.19.0 or higher"

import argparse
import os
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np

# SOURCE: https://github.com/rasbt/deeplearning-models/blob/35aba5dc03c43bc29af5304ac248fc956e1361bf/pytorch_ipynb/helper_evaluate.py
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.profiler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.transforms.functional as pytorch_transforms_functional
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


# SOURCE: https://github.com/pytorch/vision/blob/main/references/classification/train.py
def _get_cache_path(filepath) -> str:
    """_summary_

    Args:
        filepath (_type_): _description_

    Returns:
        str: _description_
    """
    import hashlib

    file_path_hash = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~",
        ".torch",
        "vision",
        "datasets",
        "imagefolder",
        file_path_hash[:10] + ".pt",
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def get_pil_image_channels(image_path: str) -> int:
    """Open an image and get the number of channels it has.

    Args:
        image_path (str): _description_

    Returns:
        int: _description_
    """
    # load pillow image
    pil_img = Image.open(image_path)

    # Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
    pil_img_tensor = transforms.PILToTensor()(pil_img)

    return pil_img_tensor.shape[0]


def convert_pil_image_to_rgb_channels(image_path: str):
    """Convert Pil image to have the appropriate number of color channels

    Args:
        image_path (str): _description_

    Returns:
        _type_: _description_
    """
    return (
        Image.open(image_path).convert("RGB")
        if get_pil_image_channels(image_path) != 4
        else Image.open(image_path)
    )


# convert image back and forth if needed: https://stackoverflow.com/questions/68207510/how-to-use-torchvision-io-read-image-with-image-as-variable-not-stored-file
def convert_pil_image_to_torch_tensor(pil_image: Image) -> torch.Tensor:
    """Convert PIL image to pytorch tensor

    Args:
        pil_image (PIL.Image): _description_

    Returns:
        torch.Tensor: _description_
    """
    return pytorch_transforms_functional.to_tensor(pil_image)


# convert image back and forth if needed: https://stackoverflow.com/questions/68207510/how-to-use-torchvision-io-read-image-with-image-as-variable-not-stored-file
def convert_tensor_to_pil_image(tensor_image: torch.Tensor) -> Image:
    """Convert tensor image to Pillow object

    Args:
        tensor_image (torch.Tensor): _description_

    Returns:
        PIL.Image: _description_
    """
    return pytorch_transforms_functional.to_pil_image(tensor_image)


def from_pil_image_to_plt_display(
    img: Image,
    pred_dicts: List[Dict],
    to_disk: bool = True,
    interactive: bool = True,
    fname: str = "plot.png",
) -> None:
    """Take a PIL image and convert it into a matplotlib figure that has the prediction along with image displayed.

    Args:
        img (Image): _description_
        image_class (str): _description_
        to_disk (bool, optional): _description_. Defaults to True.
        interactive (bool, optional): _description_. Defaults to True.
    """
    # Turn the image into an array
    img_as_array = np.asarray(img)

    image_class = pred_dicts[0]["pred_class"]
    image_pred_prob = pred_dicts[0]["pred_prob"]
    image_time_for_pred = pred_dicts[0]["time_for_pred"]

    if interactive:
        plt.ion()

    # Plot the image with matplotlib
    plt.figure(figsize=(10, 7))
    plt.imshow(img_as_array)
    title_font_dict = {"fontsize": "10"}
    plt.title(
        f"Image class: {image_class} | Image Pred Prob: {image_pred_prob} | Prediction time: {image_time_for_pred} | Image shape: {img_as_array.shape} -> [height, width, color_channels]",
        fontdict=title_font_dict,
    )
    plt.axis(False)

    if to_disk:
        # plt.imsave(fname, img_as_array)
        plt.savefig(fname)


def create_writer(
    experiment_name: str,
    model_name: str,
    extra: str = None,
) -> SummaryWriter:
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    import os
    from datetime import datetime

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime(
        "%Y-%m-%d",
    )  # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def show_confusion_matrix_helper(
    cmat: np.ndarray,
    class_names: List[str],
    to_disk: bool = True,
    fname: str = "plot.png",
) -> None:
    """_summary_

    Args:
        cmat (np.ndarray): _description_
        class_names (List[str]): _description_
        to_disk (bool, optional): _description_. Defaults to True.
        fname (str, optional): _description_. Defaults to "plot.png".
    """
    # boss: function via https://colab.research.google.com/github/mrdbourke/pytorch-deep-learning/blob/main/03_pytorch_computer_vision.ipynb#scrollTo=7aed6d76-ad1c-429e-b8e0-c80572e3ebf4
    fig, ax = plot_confusion_matrix(
        conf_mat=cmat,
        class_names=class_names,
        norm_colormap=matplotlib.colors.LogNorm(),
        # normed colormaps highlight the off-diagonals
        # for high-accuracy models better
    )

    if to_disk:
        ic("Writing confusion matrix to disk ...")
        ic(plt.savefig(fname))
    else:
        plt.show()


def compute_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
):
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        data_loader (torch.utils.data.DataLoader): _description_
        device (str): _description_

    Returns:
        _type_: _description_
    """
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def compute_epoch_loss(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
) -> Any | float | torch.Tensor:
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        data_loader (torch.utils.data.DataLoader): _description_
        device (str): _description_

    Returns:
        Any | float | torch.Tensor: _description_
    """
    model.eval()
    curr_loss, num_examples = 0.0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            loss = F.cross_entropy(logits, targets, reduction="sum")
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def compute_confusion_matrix(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device,
):
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        data_loader (torch.utils.data.DataLoader): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """

    all_targets, all_predictions = [], []
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            all_targets.extend(targets.to("cpu"))
            all_predictions.extend(predicted_labels.to("cpu"))

    all_predictions = all_predictions
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    class_labels = np.unique(np.concatenate((all_targets, all_predictions)))
    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([0, class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])
    n_labels = class_labels.shape[0]
    z = list(zip(all_targets, all_predictions))
    lst = [z.count(combi) for combi in product(class_labels, repeat=2)]
    return np.asarray(lst)[:, None].reshape(n_labels, n_labels)


def run_confusion_matrix(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
) -> None:
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        test_dataloader (torch.utils.data.DataLoader): _description_
        device (torch.device): _description_
        class_names (List[str]): _description_
    """

    cmat = compute_confusion_matrix(model, test_dataloader, device)

    # cmat, type(cmat)

    show_confusion_matrix_helper(cmat, class_names)


# def run_validate(
#     model: torch.nn.Module,
#     test_dataloader: torch.utils.data.DataLoader,
#     device: torch.device,
#     loss_fn: torch.nn.Module,
# ):
#     print(" Running in evaluate mode ...")

#     start_time = timer()
#     # Setup testing and save the results
#     test_loss, test_acc = engine.test_step(
#         model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
#     )

#     # End the timer and print out how long it took
#     end_time = timer()
#     print(f"[INFO] Total testing time: {end_time-start_time:.3f} seconds")
#     ic(test_loss)
#     ic(test_acc)


# # SOURCE: https://github.com/mrdbourke/pytorch-apple-silicon/blob/main/01_cifar10_tinyvgg.ipynb
# def write_training_results_to_csv(
#     MACHINE,
#     device,
#     dataset_name="",
#     num_epochs="",
#     batch_size="",
#     image_size="",
#     train_data="",
#     test_data="",
#     total_train_time="",
#     model="",
# ):
#     # Create results dict
#     results = {
#         "machine": MACHINE,
#         "device": device,
#         "dataset_name": dataset_name,
#         "epochs": num_epochs,
#         "batch_size": batch_size,
#         "image_size": image_size[0],
#         "num_train_samples": len(train_data),
#         "num_test_samples": len(test_data),
#         "total_train_time": round(total_train_time, 3),
#         "time_per_epoch": round(total_train_time / num_epochs, 3),
#         "model": model.__class__.__name__,
#     }

#     results_df = pd.DataFrame(results, index=[0])

#     # Write CSV to file
#     if not os.path.exists("results/"):
#         os.makedirs("results/")

#     results_df.to_csv(
#         f"results/{MACHINE.lower().replace(' ', '_')}_{device}_{dataset_name}_image_size.csv",
#         index=False,
#     )


def write_predict_results_to_csv(
    pred_dicts: List[Dict],
    args: argparse.Namespace,
) -> None:
    """_summary_

    Args:
        pred_dicts (List[Dict]): _description_
        args (argparse.Namespace): _description_
    """
    # Create results dict
    pred_df = pd.DataFrame(pred_dicts)
    pred_df.drop(columns=["class_name", "correct"], inplace=True)

    # if file does not exist write header
    if not os.path.isfile(args.results):
        pred_df.to_csv(args.results, header="column_names")
    else:  # else it exists so append without writing the header
        pred_df.to_csv(args.results, mode="a", header=False)


def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Table,
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table


def console_print_table(results_df: pd.DataFrame) -> None:
    """_summary_

    Args:
        results_df (pd.DataFrame): _description_
    """
    # Initiate a Table instance to be modified
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.DOUBLE,
        expand=True,
        show_lines=True,
        show_edge=True,
        show_footer=True,
    )

    # Modify the table instance to have the data from the DataFrame
    table = df_to_table(results_df, table)

    # Update the style of the table
    table.row_styles = ["none", "dim"]
    table.box = box.SIMPLE_HEAD

    # console.print(table)


def csv_to_df(path: str) -> pd.DataFrame:
    """_summary_

    Args:
        path (str): _description_

    Returns:
        _type_: _description_
    """
    return pd.read_csv(path)


def inspect_csv_results() -> pd.DataFrame:
    """_summary_

    Returns:
        pd.DataFrame: _description_
    """
    results_paths = list(Path("results").glob("*.csv"))

    df_list = [pd.read_csv(path) for path in results_paths]
    results_df = pd.concat(df_list).reset_index(drop=True)
    # prettify(results_df)

    # # Initiate a Table instance to be modified
    # table = Table(show_header=True, header_style="bold magenta")

    # # Modify the table instance to have the data from the DataFrame
    # table = df_to_table(results_df, table)

    # # Update the style of the table
    # table.row_styles = ["none", "dim"]
    # table.box = box.SIMPLE_HEAD

    # console.print(table)
    console_print_table(results_df)
    return results_df
