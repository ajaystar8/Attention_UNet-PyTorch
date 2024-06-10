"""
This script is used to create a train-val-test split of the dataset
"""
import os
import shutil
import argparse
from glob import glob
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

# Take command line arguments
parser = argparse.ArgumentParser(description='Script to prepare train-val-test split',
                                 epilog='Happy splitting! :)')

parser.add_argument('data_dir', metavar='data_dir', help='path to dataset directory')
parser.add_argument('storage_dir', metavar='storage_dir', help='path to directory where the train-val-test split is '
                                                               'to be placed.')
parser.add_argument('--train_val_test_split', nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"),
                    default=(0.8, 0.1, 0.1),
                    help="Train, validation and test splits (should sum to 1). (default: %(default)s)")
args = parser.parse_args()

total_sum = 0
for val in args.train_val_test_split:
    total_sum += val
    if not (0 < val < 1):
        raise ValueError("Split values should be floats between 0 and 1.")
if total_sum != 1:
    raise ValueError("Train, validation and test splits must sum to 1.")

if not os.path.exists(args.storage_dir):
    # Main directory
    os.makedirs(args.storage_dir)
    # Train-Val-Test directory
    dirs = [os.path.join(args.storage_dir, "train"), os.path.join(args.storage_dir, "val"),
            os.path.join(args.storage_dir, "test")]
    for dir_name in dirs:
        os.makedirs(dir_name)
        os.makedirs(os.path.join(dir_name, "images"))
        os.makedirs(os.path.join(dir_name, "masks"))
else:
    raise FileExistsError(f"Directory already exits at {args.data_dir}.")

# Load all images from the current directory. Update the images and masks folder paths as necessary
all_image_paths = sorted(glob(os.path.join(args.data_dir, "images", "*.png")))
all_mask_paths = sorted(glob(os.path.join(args.data_dir, "masks", "*.png")))

if len(all_image_paths) == 0 or len(all_mask_paths) == 0:
    raise FileExistsError(f"Incorrect path to data directory. No images/masks found.")

if len(all_image_paths) != len(all_mask_paths):
    raise FileExistsError(f"Incorrect path to data directory.")

# isolate train-val-test image paths
train_images, val_test_images, train_masks, val_test_masks = train_test_split(all_image_paths, all_mask_paths,
                                                                              train_size=args.train_val_test_split[0],
                                                                              shuffle=True)
adjusted_val_size = args.train_val_test_split[1] / (args.train_val_test_split[1] + args.train_val_test_split[2])
val_images, test_images, val_masks, test_masks = train_test_split(val_test_images, val_test_masks,
                                                                  train_size=adjusted_val_size, shuffle=True)

# copying the train image-mask pairs as it is
for img_path, mask_path in tqdm(zip(train_images, train_masks), total=len(train_images)):
    shutil.copy(img_path, os.path.join(args.storage_dir, "train", "images", os.path.basename(img_path)))
    shutil.copy(mask_path, os.path.join(args.storage_dir, "train", "masks", os.path.basename(mask_path)))

# copying the val image-mask pairs as it is
for img_path, mask_path in tqdm(zip(val_images, val_masks), total=len(val_images)):
    shutil.copy(img_path, os.path.join(args.storage_dir, "val", "images", os.path.basename(img_path)))
    shutil.copy(mask_path, os.path.join(args.storage_dir, "val", "masks", os.path.basename(mask_path)))

# copying the test image-mask pairs as it is
for img_path, mask_path in tqdm(zip(test_images, test_masks), total=len(test_images)):
    shutil.copy(img_path, os.path.join(args.storage_dir, "test", "images", os.path.basename(img_path)))
    shutil.copy(mask_path, os.path.join(args.storage_dir, "test", "masks", os.path.basename(mask_path)))

print("Images transferred successfully")
print(f"MURA Augmented Train-Val-Test split Dataset ready and is available at {args.storage_dir}!")
