"""
Contains various utility functions for data visualization, PyTorch model training, testing and saving.
"""
import os
import random
from glob import glob
from typing import *
from PIL import Image
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

try:
    from torchsummary import summary
except ImportError as e:
    print("Failed to import torchsummary. Please install it using 'pip install torchsummary'.")
    raise e

from config import *
from model_builder import AttentionUNet


def walk_data_directory(data_directory: str):
    """
    Walks through the data directory and prints the number of folder and files. Use for directory visualization.

    Args:
        data_directory: Directory whose visualization is to be done.

    Returns:
        None. Just prints the structure.
    """
    for root, dirs, files in os.walk(data_directory):
        print(f"In {root}, {len(dirs)} directories and {len(files)} images/masks found.")


def visualize_random_images(
        data_directory: str,
        n: int = 4):
    """
    Function to randomly pick images and corresponding masks from the entire dataset and visualize them.

    Args:
        data_directory: path to the data folder.
        n: number of random image-mask pairs to pick and plot.

    Returns:
        None. Uses matplotlib to visualize the image-mask pairs.
    """
    # for reproducibility
    random.seed(42)

    # get the list of all image and mask paths
    image_paths = sorted(glob(os.path.join(data_directory, "**/images", "*.png")))
    mask_paths = sorted(glob(os.path.join(data_directory, "**/masks", "*.png")))

    # pick a set of random indices
    random_samples_idx = random.sample(range(len(image_paths)), k=n)
    for i, idx in enumerate(random_samples_idx):
        # load the image and mask paths from the selected indices
        image, mask = Image.open(image_paths[idx]), Image.open(mask_paths[idx]).convert('L')

        # Canvas for plotting
        fig, axs = plt.subplots(1, 2)

        # Plot the image
        axs[0].imshow(image, cmap="gray")
        axs[0].set_title(f"Image\nShape:{image.size}\nMode: {image.mode}")
        axs[0].axis("off")

        # Plot the mask
        axs[1].imshow(mask, cmap="gray")
        axs[1].set_title(f"Mask\nShape:{mask.size}\nMode: {mask.mode}")
        axs[1].axis("off")

        plt.show()


def visualize_image_from_dataloader(dataloader: DataLoader):
    """
    Function to check the working of dataloaders. The function picks up an image from the dataloader and plots it.

    Args:
         dataloader: A DataLoader instance for the data.

    Returns:
        None. Plots the image-mask pair.
    """

    # get the image-mask pair
    img, mask = next(iter(dataloader))

    fig, axs = plt.subplots(1, 2)

    img_mod = img.squeeze().permute(1, 2, 0)
    mask_mod = mask.squeeze()

    axs[0].imshow(img_mod, cmap="gray")
    axs[0].set_title(f"Image\nSize: {img_mod.size()}")
    axs[0].axis(False)

    axs[1].imshow(mask_mod, cmap="gray")
    axs[1].set_title(f"Mask\nSize: {mask_mod.size()}")
    axs[1].axis(False)

    plt.show()


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int]):
    """
    Prints the summary of the PyTorch model created.

    Args:
        model: A PyTorch model instance which is to be summarized.
        input_size: Expected input dimensions of the image in tuple format (C, H, W)

    Returns:
        None.
    """
    return summary(model=model.to(DEVICE), input_size=input_size)


def plot_loss_accuracy_curves(results: Dict[str, List[float]]):
    """
    The function plots the loss and accuracy values against epoch number.

    Args:
        results: A dictionary containing the results obtained during model evaluation. The dictionary is of the form:
        {"model_name": str,
            "model_loss": float,
            "model_accuracy": float,
            "model_F1_score": float,
            "model_Jaccard_Index": float}

    Returns:
        None.
    """
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    train_accuracy = results["train_accuracy"]
    test_accuracy = results["test_accuracy"]

    epochs = range(len(train_loss))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def make_predictions(model: nn.Module,
                     image_path: str,
                     mask_path: str,
                     transform: transforms.Compose):
    """
    The function takes in a PyTorch model instance and makes a prediction on the image path specified. Finally, the
    prediction mask is visualized along the with input image and the corresponding ground truth mask.

    Args:
        model: A PyTorch model instance.
        image_path: Path to the image on which the segmentation is to be performed.
        mask_path: Path to the corresponding ground truth mask.
        transform: Transformations to be applied to the input image and mask pair.

    Returns:
        None. Makes prediction and plots them.
    """

    img = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    img, mask = transform(img, mask)
    img, mask = img.unsqueeze(dim=0), mask.unsqueeze(dim=0)

    model = model.to(DEVICE)

    model.eval()
    with torch.inference_mode():
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        pred_logit = model(img)
        pred_mask = (torch.sigmoid(pred_logit) > 0.5).float()

    fig, axs = plt.subplots(1, 3)

    image_mod = img.squeeze().detach().cpu().numpy()
    mask_mod = mask.squeeze().detach().cpu().numpy()
    pred_mask_mod = pred_mask.squeeze().detach().cpu().numpy()

    axs[0].imshow(image_mod, cmap="gray")
    axs[0].set_title("Image")
    axs[0].axis(False)

    axs[1].imshow(mask_mod, cmap="gray")
    axs[1].set_title("True Mask")
    axs[1].axis(False)

    axs[2].imshow(pred_mask_mod, cmap="gray")
    axs[2].set_title("Predicted Mask")
    axs[2].axis(False)

    plt.show()


def save_model(
        model: nn.Module,
        target_dir: str,
        model_ckpt_name: str
):
    """Saves a PyTorch model to a target directory.

      Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_ckpt_name: A filename for the saved model. Should include
          either ".pth" or ".pt" as the file extension.

      Example usage:
        save_model(model=model_0,
                   target_dir="models",
                   model_name="UNet.pth")
      """

    # check if target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # check if model name follows the guidelines for a checkpoint
    assert model_ckpt_name.split(".")[-1] == "pt" or model_ckpt_name.split(".")[-1] == "pth", \
        "model_name must end with .pt or .pth"

    # define complete model checkpoint path
    model_ckpt_path = os.path.join(target_dir, model_ckpt_name)

    # save model state dict
    print(f"[INFO] Saving model to: {model_ckpt_path}")
    torch.save(model.state_dict(), model_ckpt_path)


def load_model(model_ckpt_path: str, in_channels: int, out_channels: int):
    """
    The function takes in the path to a model checkpoint and returns a PyTorch model containing the trained weights.

    Args:
        model_ckpt_path: A path to the model checkpoint.

    Returns:
        A trained PyTorch model instance.
    """

    print(f"[INFO] Loading model checkpoint for prediction from: {model_ckpt_path}")

    # Create model instance
    model = AttentionUNet(in_channels, out_channels).to(DEVICE)

    # Load model state dict
    model.load_state_dict(torch.load(model_ckpt_path, map_location=DEVICE))

    return model
