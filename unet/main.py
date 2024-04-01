from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split
from os.path import join

from utils import *
from model import UNet

import pretrain
import segmentation_train as train

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


if __name__ == "__main__":
    torch.manual_seed(780962)

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

    ### Pretraining ###
    print("#" * 10 + " Pretraining " + "#" * 10 + "\n")

    criterion = InPaintingLoss()

    ## Synthetic data pretraining ##

    print("Pretraining on synthetic dataset")

    synth_ds = SynthDataset(ROOT_DIR, image_size=IMAGE_SIZE)

    pretrain_ds = PretrainingDataset(synth_ds, mask_generator=MASK_GENERATOR)
    train_ds, val_ds, test_ds = random_split(pretrain_ds, [SPLIT * SPLIT, SPLIT * (1 - SPLIT), 1 - SPLIT])
    print(f"Number of training examples: {len(train_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = MODEL_CLASS(PRETRAIN_NUM_OUT_CHANNELS).to(DEVICE)
    optim = Adam(model.parameters(), lr=LR)

    pretrain.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=PRETRAIN_MAX_NUM_EPOCHS,
        patience=PATIENCE,
    )

    print("Done\n")

    pretrain_image_output(model, test_dl, os.path.join(EXAMPLE_IMAGES_DIR, SYNTH_PRETRAIN_NAME + ".jpg"), DEVICE)

    model = model.to(dtype=torch.float32)
    torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, SYNTH_PRETRAIN_NAME + ".pt"))

    ## Kaggle dogs and cats pretraining ##

    print("Pretraining on Kaggle cats and dogs dataset")

    train_val_ds = PretrainingDataset(
        KaggleDogsAndCats(ROOT_DIR, split="train", image_size=IMAGE_SIZE), mask_generator=MASK_GENERATOR
    )
    test_ds = PretrainingDataset(
        KaggleDogsAndCats(ROOT_DIR, split="test", image_size=IMAGE_SIZE), mask_generator=MASK_GENERATOR
    )

    train_ds, val_ds = random_split(train_val_ds, [SPLIT, 1 - SPLIT])
    print(f"Number of training examples: {len(train_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = MODEL_CLASS(PRETRAIN_NUM_OUT_CHANNELS).to(DEVICE)
    optim = Adam(model.parameters(), lr=LR)

    pretrain.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=PRETRAIN_MAX_NUM_EPOCHS,
        patience=PATIENCE,
    )

    print("Done\n")

    pretrain_image_output(model, test_dl, os.path.join(EXAMPLE_IMAGES_DIR, KAGGLE_PRETRAIN_NAME + ".jpg"), DEVICE)

    model = model.to(dtype=torch.float32)
    torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, KAGGLE_PRETRAIN_NAME + ".pt"))

    ### Supervised segmentation training ###

    print("#" * 10 + " Image segmentation training " + "#" * 10 + "\n")

    train_val_ds = OxfordPetsDataset(ROOT_DIR, split="train", image_size=IMAGE_SIZE)
    test_ds = OxfordPetsDataset(ROOT_DIR, split="test", image_size=IMAGE_SIZE)
    train_ds, val_ds = random_split(train_val_ds, [SPLIT, 1 - SPLIT])
    print(f"Number of training examples: {len(train_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    criterion = nn.BCEWithLogitsLoss()

    ## No pretraining ##

    print("No pretraining")

    model = MODEL_CLASS(SEG_NUM_OUT_CHANNELS).to(DEVICE)
    optim = Adam(model.parameters(), lr=LR)

    train.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=TRAIN_MAX_NUM_EPOCHS,
        patience=PATIENCE,
    )

    test_score = model_iou(model, test_dl, DEVICE)
    print(f"Test IOU: {test_score:.4g}")

    segmentation_image_output(model, test_dl, os.path.join(EXAMPLE_IMAGES_DIR, NO_PRETRAIN_SEG_NAME + ".jpg"), DEVICE)

    model = model.to(dtype=torch.float32)
    torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, NO_PRETRAIN_SEG_NAME + ".pt"))

    print("Done\n")

    ## Kaggle pretrain segmentation ##

    print("Image segmentation train with Kaggle dogs and cats pretrain")
    model = MODEL_CLASS(PRETRAIN_NUM_OUT_CHANNELS)
    model.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, KAGGLE_PRETRAIN_NAME + ".pt")))
    model.new_head(SEG_NUM_OUT_CHANNELS)
    model = model.to(DEVICE)
    optim = Adam(model.parameters(), lr=LR)

    train.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=TRAIN_MAX_NUM_EPOCHS,
        patience=PATIENCE,
    )

    test_score = model_iou(model, test_dl, DEVICE)
    print(f"Test IOU: {test_score:.4g}")

    segmentation_image_output(model, test_dl, os.path.join(EXAMPLE_IMAGES_DIR, KAGGLE_SEG_NAME + ".jpg"), DEVICE)

    model = model.to(dtype=torch.float32)
    torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, KAGGLE_SEG_NAME + ".pt"))

    print("Done\n")

    ## Synthetic pretrain segmentation ##

    print("Image segmentation train with synthetic data pretrain")
    model = MODEL_CLASS(PRETRAIN_NUM_OUT_CHANNELS)
    model.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, SYNTH_PRETRAIN_NAME + ".pt")))
    model.new_head(SEG_NUM_OUT_CHANNELS)
    model = model.to(DEVICE)
    optim = Adam(model.parameters(), lr=LR)

    train.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=TRAIN_MAX_NUM_EPOCHS,
        patience=PATIENCE,
    )

    test_score = model_iou(model, test_dl, DEVICE)
    print(f"Test IOU: {test_score:.4g}")

    segmentation_image_output(model, test_dl, os.path.join(EXAMPLE_IMAGES_DIR, SYNTH_SEG_NAME + ".jpg"), DEVICE)

    model = model.to(dtype=torch.float32)
    torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, SYNTH_SEG_NAME + ".pt"))

    print("Done\n")
