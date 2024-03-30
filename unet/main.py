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

    batch_size = 32
    eval_batch_size = 64
    pretrain_max_num_epochs = 20
    train_max_num_epochs = 20
    patience = 5
    num_workers = 8

    model_class = UNet

    kaggle_pretrain_name = "kaggle_pretrain"
    synth_pretrain_name = "synth_pretrain"
    no_pretrain_seg_name = "no_pretrain"
    kaggle_seg_name = "kaggle_seg"
    synth_seg_name = "synth_seg"

    split = 0.8

    pretrain_num_out_channels = 3 # Reconstruction image
    seg_num_out_channels = 1 # Foreground, background segmentation
    square_size = 16
    image_size = (240, 240)
    mask_generator = CheckerboardMask(square_size=square_size, image_size=image_size)

    lr = 1e-3

    root_dir = "../data"

    ### Pretraining ###
    print("#" * 10 + " Pretraining " + "#" * 10 + "\n")

    criterion = InPaintingLoss()

    ## Synthetic data pretraing ##

    print("Pretraining on synthetic dataset")

    synth_ds = SynthDataset(root_dir, image_size=image_size)

    pretrain_ds = PretrainingDataset(synth_ds, mask_generator=mask_generator)
    train_ds, val_ds, test_ds = random_split(pretrain_ds, [split * split, split * (1 - split), 1 - split])
    print(f"Number of training examples: {len(train_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

    model = model_class(pretrain_num_out_channels).to(DEVICE)
    optim = Adam(model.parameters(), lr=lr)

    pretrain.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=pretrain_max_num_epochs,
        patience=patience,
    )

    print("Done\n")

    pretrain_image_output(model, test_dl, synth_pretrain_name + ".jpg", DEVICE)

    model = model.to(dtype=torch.float32)
    torch.save(model.state_dict(), synth_pretrain_name + ".pt")

    ## Kaggle dogs and cats pretraining ##

    print("Pretraining on Kaggle cats and dogs dataset")

    train_val_ds = PretrainingDataset(
        KaggleDogsAndCats(root_dir, split="train", image_size=image_size), mask_generator=mask_generator
    )
    test_ds = PretrainingDataset(
        KaggleDogsAndCats(root_dir, split="test", image_size=image_size), mask_generator=mask_generator
    )

    train_ds, val_ds = random_split(train_val_ds, [split, 1 - split])
    print(f"Number of training examples: {len(train_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

    model = model_class(pretrain_num_out_channels).to(DEVICE)
    optim = Adam(model.parameters(), lr=lr)

    pretrain.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=pretrain_max_num_epochs,
        patience=patience,
    )

    print("Done\n")

    pretrain_image_output(model, test_dl, kaggle_pretrain_name + ".jpg", DEVICE)

    model = model.to(dtype=torch.float32)
    torch.save(model.state_dict(), kaggle_pretrain_name + ".pt")

    ### Supervised segmentation training ###

    print("#" * 10 + " Image segmentation training " + "#" * 10 + "\n")

    train_val_ds = OxfordPetsDataset(root_dir, split="train", image_size=image_size)
    test_ds = OxfordPetsDataset(root_dir, split="test", image_size=image_size)
    train_ds, val_ds = random_split(train_val_ds, [split, 1 - split])
    print(f"Number of training examples: {len(train_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.BCEWithLogitsLoss()

    ## No pretraining ##

    print("No pretraining")

    model = model_class(seg_num_out_channels).to(DEVICE)
    optim = Adam(model.parameters(), lr=lr)

    train.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=train_max_num_epochs,
        patience=patience,
    )

    test_score = model_iou(model, test_dl, DEVICE)
    print(f"Test IOU: {test_score:.4g}")

    segmentation_image_output(model, test_dl, no_pretrain_seg_name + ".jpg", DEVICE)

    model = model.to(dtype=torch.float32)
    torch.save(model.state_dict(), no_pretrain_seg_name + ".pt")

    print("Done\n")

    ## Kaggle pretrain segmentation ##

    print("Image segmentation train with Kaggle dogs and cats pretrain")
    model = model_class(pretrain_num_out_channels)
    model.load_state_dict(torch.load(kaggle_pretrain_name + ".pt"))
    model.new_head(seg_num_out_channels)
    model = model.to(DEVICE)
    optim = Adam(model.parameters(), lr=lr)

    train.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=train_max_num_epochs,
        patience=patience,
    )

    test_score = model_iou(model, test_dl, DEVICE)
    print(f"Test IOU: {test_score:.4g}")

    segmentation_image_output(model, test_dl, kaggle_seg_name + ".jpg", DEVICE)

    model = model.to(dtype=torch.float32)
    torch.save(model.state_dict(), kaggle_seg_name + ".pt")

    print("Done\n")

    ## Synthetic pretrain segmentation ##

    print("Image segmentation train with synthetic data pretrain")
    model = model_class(pretrain_num_out_channels)
    model.load_state_dict(torch.load(synth_pretrain_name + ".pt"))
    model.new_head(seg_num_out_channels)
    model = model.to(DEVICE)
    optim = Adam(model.parameters(), lr=lr)

    train.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=train_max_num_epochs,
        patience=patience,
    )

    test_score = model_iou(model, test_dl, DEVICE)
    print(f"Test IOU: {test_score:.4g}")

    segmentation_image_output(model, test_dl, synth_seg_name + ".jpg", DEVICE)

    model = model.to(dtype=torch.float32)
    torch.save(model.state_dict(), synth_seg_name + ".pt")

    print("Done\n")
