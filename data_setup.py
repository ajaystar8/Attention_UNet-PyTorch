"""
Contains functionality for creating PyTorch Datasets and DataLoaders for semantic image segmentation data.
"""

import os.path
from glob import glob
from typing import *
from PIL import Image

import torch
import torchvision.transforms.v2 as transforms

from torch.utils.data import Dataset, DataLoader


class SegmentationDataset(Dataset):
    """
    A generic base class for storing datasets for semantic segmentation tasks.
    """

    def __init__(self, image_paths, mask_paths, transform=None) -> None:

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # load image
        img = Image.open(self.image_paths[idx]).convert("L")

        # load mask
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # apply transforms if given and return the (img, mask) pair
        if self.transform:
            return self.transform(img, mask)
        else:
            return img, mask

    def __len__(self) -> int:
        return len(self.image_paths)


def create_dataloaders(
        train_dir: str,
        val_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
):
    """
    Creates training and testing DataLoaders.

    Takes in training and testing directory paths and turns them into PyTorch Datasets and then into PyTorch DataLoaders

    Args:
        train_dir: Path to the training directory.
        val_dir: Path to the validation directory
        test_dir: Path to the test directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch to in each of the DataLoaders.

    Returns:
        A tuple of (train_dataloader, test_dataloader).
        Example usage:
            train_dataloader, test_dataloader = create_dataloaders(
                train_dir=path/to/train/dir,
                test_dir=path/to/test/dir,
                transform=some_transform,
                batch_size=1
            )
    """
    train_image_paths = sorted(glob(os.path.join(train_dir, "images", "*.png")))
    train_mask_paths = sorted(glob(os.path.join(train_dir, "masks", "*.png")))

    val_image_paths = sorted(glob(os.path.join(val_dir, "images", "*.png")))
    val_mask_paths = sorted(glob(os.path.join(val_dir, "masks", "*.png")))

    test_image_paths = sorted(glob(os.path.join(test_dir, "images", "*.png")))
    test_mask_paths = sorted(glob(os.path.join(test_dir, "masks", "*.png")))

    # Create train dataset
    train_data = SegmentationDataset(image_paths=train_image_paths,
                                     mask_paths=train_mask_paths,
                                     transform=transform)

    val_data = SegmentationDataset(image_paths=val_image_paths,
                                    mask_paths=val_mask_paths,
                                    transform=transform)

    # Create test dataset
    test_data = SegmentationDataset(image_paths=test_image_paths,
                                    mask_paths=test_mask_paths,
                                    transform=transform)

    # Create dataloaders for the training and testing datasets

    # train dataloader
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                shuffle=True)

    # test dataloader
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
