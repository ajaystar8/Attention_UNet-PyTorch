"""
Contains functions for training, validating and testing a PyTorch model.
"""
import os
import time
from config import *
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils.utils import save_model, load_model

import wandb


def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: optim.Optimizer,
               dice_fn, precision_fn, recall_fn):
    """
    Train a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then runs through all the required training steps, viz.,
     forward pass, loss calculation, optimizer step.

     Args:
        model: A pyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A Pytorch loss function to minimize.
        optimizer: A Pytorch optimizer to help minimize the loss function.
        dice_fn: A custom function to calculate the dice score.
        precision_fn: A torchmetrics instance to measure precision.
        recall_fn: A torchmetrics instance to measure recall.

    Returns:
        A dictionary containing the validation metrics in the form of:
        {"train_loss": train_loss, "train_dice": train_dice, "train_precision": train_precision,
            "train_recall": train_recall}
    """

    # send model to device and put it in train mode
    model.to(DEVICE)
    model.train()

    train_loss, train_dice, train_precision, train_recall = 0, 0, 0, 0
    for X, y in dataloader:
        # send data to device
        X, y = X.to(DEVICE), y.to(DEVICE)

        # 1. Forward Pass
        y_logits = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_logits, y)
        train_loss += loss

        # Calculate performance metrics
        train_dice += dice_fn(torch.round(torch.sigmoid(y_logits)).to(torch.float), torch.round(y).to(torch.float))
        train_precision += precision_fn(y_logits, torch.round(y))
        train_recall += recall_fn(y_logits, torch.round(y))

        # 3. Clear optimizer gradients
        optimizer.zero_grad()

        # 4. Backward Pass
        loss.backward()

        # 5. Update weights
        optimizer.step()

    # Adjust the metrics to get the average per batch
    train_loss /= len(dataloader)
    train_dice /= len(dataloader)
    train_precision /= len(dataloader)
    train_recall /= len(dataloader)

    print(f"\nTrain Loss: {train_loss:.5f} | Train DSC: {train_dice:.5f} | Train Precision: {train_precision:.5f} "
          f"| Train Recall: {train_recall:.5f}")
    return {"train_loss": train_loss, "train_dice": train_dice, "train_precision": train_precision,
            "train_recall": train_recall}


def val_step(model: nn.Module,
             dataloader: DataLoader,
             loss_fn: nn.Module,
             dice_fn, precision_fn, recall_fn):
    """
    Validate a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs a forward pass on a validation dataset.

    Args:
        model: A PyTorch model to be validated.
        dataloader: A DataLoader instance for the model to be validated on.
        loss_fn: A PyTorch loss function to calculate loss on validated data.
        dice_fn: A custom function to calculate the dice score.
        precision_fn: A torchmetrics instance to measure precision.
        recall_fn: A torchmetrics instance to measure recall.

    Returns:
        A dictionary containing the validation metrics in the form of:
        {"val_loss": val_loss, "val_dice": val_dice, "val_precision": val_precision, "val_recall": val_recall}
    """

    # send model to device
    model.to(DEVICE)

    val_loss, val_dice, val_precision, val_recall = 0, 0, 0, 0

    # set model to eval mode
    model.eval()
    with torch.inference_mode():
        # Loop through the DataLoader batches
        for X, y in dataloader:
            # send device to device
            X, y = X.to(DEVICE), y.to(DEVICE)

            # 1. Forward Pass
            y_logits = model(X)

            # 2. Calculate loss and performance metrics
            val_loss += loss_fn(y_logits, torch.round(y))
            val_dice += dice_fn(torch.round(torch.sigmoid(y_logits)).to(torch.int), torch.round(y).to(torch.int))
            val_precision += precision_fn(y_logits, torch.round(y))
            val_recall += recall_fn(y_logits, torch.round(y))

        # Adjust metrics by calculating average
        val_loss /= len(dataloader)
        val_dice /= len(dataloader)
        val_precision /= len(dataloader)
        val_recall /= len(dataloader)

        print(f"\nVal Loss: {val_loss:.5f} | Val DSC: {val_dice:.5f}| "
              f"Val Precision: {val_precision:.5f} | Val Recall: {val_recall:.5f}\n")
        return {"val_loss": val_loss, "val_dice": val_dice, "val_precision": val_precision, "val_recall": val_recall}


def train(model: nn.Module,
          epochs: int,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          loss_fn: nn.Module,
          optimizer: optim.Optimizer,
          dice_fn, precision_fn, recall_fn, model_ckpt_name: str, checkpoint_dir: str, verbose: int):
    """
    Trains and validates a PyTorch model.

    Passes a target PyTorch model, through train_step() and val_step() functions for a specified number of epochs,
    training and validating the model in the same epoch loop. If the validation Dice-score of the model improves,
    the function save the model's state_dict as a .pth file.

    Args:
        model: A PyTorch model to be trained and validated.
        train_dataloader: A DataLoader instance for training data.
        val_dataloader: A DataLoader instance for validation data.
        optimizer: A PyTorch optimizer.
        loss_fn: A PyTorch loss function.
        dice_fn: A custom function to calculate the dice score.
        precision_fn: A torchmetrics instance to measure precision.
        recall_fn: A torchmetrics instance to measure recall.
        checkpoint_dir: path to directory where the model checkpoints are stored

    Returns:
        A tuple of containing train and validation metrics (in form of dict) in form of:
            (train_metrics, val_metrics)
    """

    print("[INFO] Training started...")

    # create empty results dictionary
    train_metrics = {
        "train_loss": [], "train_dice": [], "train_precision": [], "train_recall": [],
    }

    val_metrics = {
        "val_loss": [], "val_dice": [], "val_precision": [], "val_recall": []
    }

    # Track the best obtained Dice Score
    # Loop through training and validation steps for a number of epochs
    max_validation_dice, max_train_time, max_val_time = 0.0000, 0, 0
    for epoch in tqdm(range(epochs)):

        print(f"\nEPOCH-{epoch + 1}------------------------------------------------------\n")

        train_start_time = time.time()
        train_epoch_metrics = train_step(model, train_dataloader, loss_fn, optimizer, dice_fn, precision_fn, recall_fn)
        train_end_time = time.time()
        total_train_time = round(train_end_time - train_start_time)

        val_start_time = time.time()
        val_epoch_metrics = val_step(model, val_dataloader, loss_fn, dice_fn, precision_fn, recall_fn)
        val_end_time = time.time()
        total_val_time = round(val_end_time - val_start_time)

        max_train_time = max(max_train_time, total_train_time)
        max_val_time = max(max_val_time, total_val_time)

        print(f"\n[INFO] Train time: {total_train_time}s\n"
              f"[INFO] Inference time: {total_val_time}s\n")

        val_dice = val_epoch_metrics["val_dice"]
        if val_dice > max_validation_dice:
            print(f"\nModel performance improved from Dice Score of {max_validation_dice:.5f} to "
                  f"Dice Score of {val_dice:.5f}\n")
            print("[INFO] Saving model checkpoint...")
            save_model(model=model, target_dir=checkpoint_dir, model_ckpt_name=model_ckpt_name)
            max_validation_dice = val_dice

        metrics = ["loss", "dice", "precision", "recall"]
        for i, metric in enumerate(metrics):
            train_metrics[f"train_{metric}"].append(train_epoch_metrics[f"train_{metric}"].cpu().detach().numpy())
            val_metrics[f"val_{metric}"].append(val_epoch_metrics[f"val_{metric}"].cpu().detach().numpy())

        wandb.log({**train_epoch_metrics, **val_epoch_metrics})

        if verbose == 1:
            wandb.alert(
                title=f"Epoch-{epoch + 1} completed!",
                text=f"Val-Dice-Score: {val_epoch_metrics['val_dice']:.3f} \n",
                level=wandb.AlertLevel.INFO,
            )

    wandb.log({"train_time": max_train_time, "val_time": max_val_time})
    wandb.finish()
    return train_metrics, val_metrics


def test_model(model_ckpt_name: str,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               dice_fn, precision_fn, recall_fn, checkpoint_dir: str,
               in_channels: int, out_channels: int):
    """
    Stores and returns the performance metrics when the model is tested on the testing dataset. Has similar
    functionality to the val_step function above.

    Args:
        model_ckpt_name: name of the PDR-UNet checkpoint to be loaded and tested.
        dataloader: A DataLoader instance.
        loss_fn: A PyTorch loss function.
        dice_fn: A custom function to calculate the dice score.
        precision_fn: A torchmetrics instance to measure precision.
        recall_fn: A torchmetrics instance to measure recall.
        checkpoint_dir: path to directory where the model checkpoints are stored
        in_channels: number of channels in the input image (default 1)
        out_channels: number of classes in ground truth mask (default 1)


    Returns:
        The test result metrics of the model under consideration in the form of:
            {"test_loss": test_loss, "test_dice": test_dice, "test_precision": test_precision,
               "test_recall": test_recall}
    """
    model = load_model(os.path.join(checkpoint_dir, model_ckpt_name), in_channels, out_channels)
    model.to(DEVICE)

    test_loss, test_dice, test_precision, test_recall = 0, 0, 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            y_logits = model(X)

            test_loss += loss_fn(y_logits, torch.round(y))
            test_dice += dice_fn(torch.round(torch.sigmoid(y_logits)).to(torch.int), torch.round(y).to(torch.int))
            test_precision += precision_fn(y_logits, torch.round(y))
            test_recall += recall_fn(y_logits, torch.round(y))

        test_loss /= len(dataloader)
        test_dice /= len(dataloader)
        test_precision /= len(dataloader)
        test_recall /= len(dataloader)

    test_results = {"test_loss": test_loss, "test_dice": test_dice, "test_precision": test_precision,
                    "test_recall": test_recall}
    return test_results
