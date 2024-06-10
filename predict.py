"""
Contains the functionality to load a model checkpoint, make a prediction for a random image from the testing set
and finally visualize the prediction.

Change the name of the model checkpoint name in the model_path according to your needs.
"""
import argparse
import random
from glob import glob
import torchvision.transforms.v2 as transforms

import os
import config
from utils.utils import make_predictions, load_model

# Take command line arguments
parser = argparse.ArgumentParser(description='Script to segment an image using a trained checkpoint of Attention-UNet.',
                                 epilog='Happy segmenting! :)')

parser.add_argument('data_dir', metavar='data_dir', help='path to dataset directory')
parser.add_argument('checkpoint_dir', metavar='checkpoint_dir',
                                  help='path to directory storing model checkpoints')

parser.add_argument('--input_dims', nargs=2, type=int, metavar=("H", "W"),
                                   help="spatial dimensions of input image (default: %(default)s)", default=[256, 256])
parser.add_argument('--in_channels', metavar="IN_C", type=int,
                                       help='number of channels in input image (default: %(default)s)', default=1)
parser.add_argument('--out_channels', metavar="OUT_C", type=int,
                                       help='number of classes in ground truth mask (default: %(default)s)', default=1)

args = parser.parse_args()

# For reproducibility
random.seed(20)

# random image path
img_path = random.choice(list(glob(os.path.join(os.path.join(args.data_sir, "test"), "images", "*.png"))))

# corresponding mask path
mask_path = os.path.join(os.path.join(os.path.join(os.path.join(args.data_sir, "test"), "masks",
                                                   os.path.basename(img_path))))

# model checkpoint path
model_path = os.path.join(args.checkpoint_dir, "attention_unet.pth")

# basic transforms to be applied to image for the model to make the forward pass properly.
simple_transforms = transforms.Compose([
    transforms.Resize(size=(args.input_dims[0], args.input_dims[1])),
    transforms.ToTensor()
])

# load trained PyTorch model instance
baseline_0 = load_model(model_ckpt_path=model_path, in_channels=args.in_channels, out_channels=args.out_channels)

# send model to device for predictions
baseline_0 = baseline_0.to(config.DEVICE)

# get_model_summary(baseline_0)

# Make the prediction
print(f"Making prediction for ({img_path}) having label mask at ({mask_path})")
make_predictions(baseline_0, img_path, mask_path, simple_transforms)
