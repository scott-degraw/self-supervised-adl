import torch
import torch.nn as nn

from utils import *
from model import UNet

assert torch.cuda.is_available(), "CUDA support is required"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
DTYPE = torch.float32

class DummyModel(nn.Module):
    """
    Dummy model to check training script works. Has single convolutional layer.
    """

    def __init__(self, num_out_channels):
        super().__init__()
        self.num_out_channels = num_out_channels
        self.new_head(num_out_channels)

    def new_head(self, num_out_channels):
        self.head = nn.Conv2d(3, num_out_channels, kernel_size=1)

    def __call__(self, images: torch.Tensor):
        return self.head(images)

BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
PRETRAIN_MAX_NUM_EPOCHS = 1
TRAIN_MAX_NUM_EPOCHS = 1
PATIENCE = 5
NUM_WORKERS = 8

MODEL_CLASS = UNet

KAGGLE_PRETRAIN_NAME = "kaggle_pretrain"
SYNTH_PRETRAIN_NAME = "synth_pretrain"
NO_PRETRAIN_SEG_NAME = "no_pretrain"
KAGGLE_SEG_NAME = "kaggle_seg"
SYNTH_SEG_NAME = "synth_seg"

SPLIT = 0.8

PRETRAIN_NUM_OUT_CHANNELS = 3 # Reconstruction image
SEG_NUM_OUT_CHANNELS = 1 # Foreground, background segmentation
SQUARE_SIZE = 16
IMAGE_SIZE = (240, 240)
MASK_GENERATOR = CheckerboardMask(square_size=SQUARE_SIZE, image_size=IMAGE_SIZE)

LR = 1e-3

ROOT_DIR = "../data"
SAVED_MODEL_DIR = "../saved_models"
EXAMPLE_IMAGES_DIR = "../example_images"
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
os.makedirs(EXAMPLE_IMAGES_DIR, exist_ok=True)