"""
Trains a PyTorch semantic segmentation model using device-agnostic code.
"""
import os
import argparse
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms.v2 as transforms
import wandb
from metrics import dice_coefficient

try:
    # noinspection PyUnresolvedReferences
    from torchmetrics.classification import BinaryPrecision, BinaryRecall
except ImportError as e:
    print("Failed to import torchmetrics. Please install it using `pip install torchmetrics`")

import config
from engine import train, test_model
from model_builder import AttentionUNet
from data_setup import create_dataloaders

# Take command line arguments
parser = argparse.ArgumentParser(description='Script to begin training and validation of Attention-UNet.',
                                 epilog='Happy training! :)')

# positional arguments
parser.add_argument('data_dir', metavar='DATA_DIR', help='path to dataset directory')
parser.add_argument('checkpoint_dir', metavar='CHECKPOINT_DIR',
                                  help='path to directory storing model checkpoints')
parser.add_argument('run_name', metavar='RUN_NAME', help='Name of current run')
parser.add_argument('dataset_name', metavar='DATASET_NAME',
                    help='Name of dataset over which model is to be trained')
parser.add_argument('wandb_api_key', metavar='WANDB_API_KEY',
                    help='API key of your Weights and Biases Account.')

# optional arguments
parser.add_argument('-v', '--verbose', type=int, metavar='VERBOSITY', choices=[0, 1], default=0,
                    help="setting verbosity to 1 will send email alerts to user after every epoch "
                         "(default: %(default)s)")
hyperparameters_group = parser.add_argument_group("Hyperparameters for model training")
hyperparameters_group.add_argument('--input_dims', nargs=2, type=int, metavar=("H", "W"),
                                   help="spatial dimensions of input image (default: %(default)s)", default=[256, 256])
hyperparameters_group.add_argument('--epochs', type=int, metavar='NUM_EPOCHS',
                                   help='number of epochs to train (default: %(default)s)', default=10)
hyperparameters_group.add_argument('--batch_size', type=int, metavar='N',
                                   help='number of images per batch (default: %(default)s)', default=1)
hyperparameters_group.add_argument('--learning_rate', type=float, metavar='LR',
                                   help='learning rate for training (default: %(default)s)', default=1e-4)

architecture_params_group = parser.add_argument_group("Architecture parameters")
architecture_params_group.add_argument('--in_channels', metavar="IN_C", type=int,
                                       help='number of channels in input image (default: %(default)s)', default=1)
architecture_params_group.add_argument('--out_channels', metavar="OUT_C", type=int,
                                       help='number of classes in ground truth mask (default: %(default)s)', default=1)

args = parser.parse_args()

# setup wandb
# comment this line out, if you want to permanently set your API Keys in config/private_keys.py
wandb.login(key=args.wandb_api_key)

# Transforms to convert the image in the format expected by the model
simple_transforms = transforms.Compose([
    transforms.Resize(size=(args.input_dims[0], args.input_dims[1])),
    transforms.ToTensor(),
])

# Generate dataloaders
train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
    train_dir=os.path.join(args.data_dir, 'train'),
    val_dir=os.path.join(args.data_dir, 'val'),
    test_dir=os.path.join(args.data_dir, 'test'),
    transform=simple_transforms,
    batch_size=args.batch_size
)

# create model instance
model = AttentionUNet(in_channels=args.in_channels, out_channels=args.out_channels).to(config.DEVICE)

# get_model_summary(baseline_0)

# create a loss function instance
loss_fn = nn.BCEWithLogitsLoss()

# create an optimizer instance
optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)

# Custom created function to calculate dice score
dice_fn = dice_coefficient

# torchmetrics instances to calculate precision and recall
precision_fn = BinaryPrecision().to(config.DEVICE)
recall_fn = BinaryRecall().to(config.DEVICE)

MODEL_CKPT_NAME = "attention_unet.pth"

config = {
    "image_size": (args.in_channels, args.input_dims[0], args.input_dims[1]),
    "dataset": args.dataset_name,
    "sample_size": len(train_dataloader) + len(val_dataloader) + len(test_dataloader),
    "train_val_test_split": "{}-{}-{}".format(config.SPLIT["train"], config.SPLIT["val"], config.SPLIT["test"]),
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "model": model.__class__.__name__,
    "learning_rate": args.learning_rate,
    "loss_fn": loss_fn.__class__.__name__,
    "optimizer": optimizer.__class__.__name__
}

# initialize a wandb run
run = wandb.init(
    project="Attention_UNet",
    name=args.run_name,
    config=config,
)

# define metrics
wandb.define_metric("train_dice", summary="max")
wandb.define_metric("val_dice", summary="max")

wandb.define_metric("train_precision", summary="max")
wandb.define_metric("val_precision", summary="max")

wandb.define_metric("train_recall", summary="max")
wandb.define_metric("val_recall", summary="max")

# copy your config
experiment_config = wandb.config

# For tracking gradients
wandb.watch(model, log="gradients", log_freq=1)

# training
wandb.alert(
    title="Training started",
    text=args.run_name,
    level=wandb.AlertLevel.INFO,
)

# Perform model training
baseline_0_train_results = train(
    model=model,
    epochs=args.epochs,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    dice_fn=dice_fn, precision_fn=precision_fn, recall_fn=recall_fn,
    model_ckpt_name=MODEL_CKPT_NAME, checkpoint_dir=args.checkpoint_dir, verbose=args.verbose
)

# Perform testing on the trained model
baseline_0_results = test_model(
    model_ckpt_name=MODEL_CKPT_NAME,
    dataloader=test_dataloader,
    loss_fn=loss_fn,
    dice_fn=dice_fn, precision_fn=precision_fn, recall_fn=recall_fn, checkpoint_dir=args.checkpoint_dir,
    in_channels=args.in_channels, out_channels=args.out_channels
)
