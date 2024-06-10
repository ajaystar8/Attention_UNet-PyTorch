import torch

SPLIT = {"train": 80, "val": 10, "test": 10}
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

