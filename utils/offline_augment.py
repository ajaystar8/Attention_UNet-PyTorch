import argparse
import os
from PIL import Image
from glob import glob
from tqdm.auto import tqdm
import torchvision.transforms.v2 as transforms

# Take command line arguments
parser = argparse.ArgumentParser(description='Script to perform offline data augmentation using PyTorch.',
                                 epilog='Happy augmenting! :)')

# positional arguments
parser.add_argument('data_dir', metavar='DATA_DIR', help='path to dataset directory consisting of non-augmented '
                                                         'images-masks pairs')
parser.add_argument('storage_dir', metavar='STORAGE_DIR', help='path to directory where the augmented image-mask '
                                                               'pairs are to be stored')
parser.add_argument('num_variations', metavar='NUM_VARIATIONS', type=int, help='number of variations to be created '
                                                                               'per image')

args = parser.parse_args()

# create storage directory if it does not exist
if not os.path.exists(args.storage_dir):
    os.makedirs(args.storage_dir)
    os.makedirs(os.path.join(args.storage_dir, 'images'))
    os.makedirs(os.path.join(args.storage_dir, 'masks'))
    print(f'Storage directory created successfully at {args.storage_dir}.')
else:
    raise FileExistsError(f"The storage directory already exists at {args.storage_dir}.")

# consolidate all image-mask paths
all_image_paths = sorted(glob(os.path.join(args.data_dir, "images", "*.png")))
all_mask_paths = sorted(glob(os.path.join(args.data_dir, "masks", "*.png")))

# Check for mismatched image-mask pairs
for img_path, mask_path in zip(all_image_paths, all_mask_paths):
    if os.path.basename(img_path) != os.path.basename(mask_path):
        raise ValueError(f"image path: {img_path} not matching with mask path: {mask_path}")

# define the transforms to be used for offline augmentation
offline_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=45, translate=(0.1, 0.3), scale=(0.5, 0.75))
])

# perform augmentation and save the augmented images
# iterate through all images-mask pairs
for img_path, mask_path in tqdm(zip(all_image_paths, all_mask_paths), total=len(all_image_paths)):
    # read image-mask pair
    img, mask = Image.open(img_path).convert("L"), Image.open(mask_path).convert("L")
    # create 25 variations of each image
    for i in range(args.num_variations):
        # perform augmentation
        img_mod, mask_mod = offline_transforms(img, mask)
        # name the image/mask pair
        title = os.path.basename(img_path).split('.')[0] + f"_aug_{i + 1}.png"
        # save image
        img_mod.save(os.path.join(args.storage_dir, "images", title))
        # save mask
        mask_mod.save(os.path.join(args.storage_dir, "masks", title))

print("Offline augmentations completed successfully.")
