""" 
Semantic segmentation training and results.
"""

from os.path import join

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.cuda.amp import autocast

from torch.utils.data import random_split

from utils import *
from run_config import *


def epoch_step(
    train_dl: torch.utils.data.DataLoader, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer
) -> float:
    """
    Do one epoch training for semantic segmentation.

    Inputs:
        train_dl (data.DataLoader): Training dataloader.
        model (nn.Module): Model to train.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
    Returns:
        total_loss (float): Mean loss for the epoch.
    """
    total_loss = 0.0
    model.train()
    for inputs, targets in train_dl:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass
        optimizer.zero_grad()
        # Use half precision for training.
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += inputs.shape[0] * loss.item()

        # Backward pass and Optimize
        SCALER.scale(loss).backward()
        SCALER.step(optimizer)
        SCALER.update()

    return total_loss / len(train_dl.dataset)

def train_loop(
    train_dl: DataLoader,
    val_dl: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optim: torch.optim.Optimizer,
    max_num_epochs: int = 10,
):
    """
    Training loop for semantic segmentation. Trains the inputted model.

    Args:
        train_dl (DataLoader): Training dataloader.
        val_dl (DataLoader): Validation dataloader.
        model (nn.Module): Segmentation model to train.
        criterion (nn.Module): Loss function.
        optim (torch.optim.Optimizer): Optimiser.
        max_num_epochs (int, optional): Maximum number of epochs to train for. Defaults to 10.
    """
    val_scores = [] # Validation scores for each epoch

    model_state_dicts = [] # Saved model state_dicts for each epoch

    for epoch in range(max_num_epochs):
        # Train
        epoch_loss = epoch_step(train_dl, model, criterion, optim)
        # Test
        val_score = model_iou(model, val_dl, DEVICE)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4g} Val IOU: {val_score:.4g}")

        val_scores.append(val_score)

        cpu_state_dict = {key: tensor.cpu() for key, tensor in model.state_dict().items()}
        model_state_dicts.append(cpu_state_dict)

    # Choose the best epoch and load that into the model.
    best_epoch = torch.tensor(val_scores).argmax()
    model.load_state_dict(model_state_dicts[best_epoch])

if __name__ == "__main__":
    torch.manual_seed(1537890)

    print("#" * 10 + " Image segmentation training " + "#" * 10 + "\n")

    full_train_val_ds = OxfordPetsDataset(ROOT_DIR, split="train", image_size=IMAGE_SIZE)
    test_ds = OxfordPetsDataset(ROOT_DIR, split="test", image_size=IMAGE_SIZE)

    # Split the training set into random subsets.
    train_sample_splits = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]

    # Perform 5 runs
    n_runs = 5

    for run in range(n_runs):
        print(f"##### run {run + 1} #####")
        for train_split in train_sample_splits:
            if train_split != 1.0:
                train_val_ds, _ = random_split(full_train_val_ds, [train_split, 1 - train_split])
            else:
                train_val_ds = full_train_val_ds

            train_ds, val_ds = random_split(train_val_ds, [SPLIT, 1 - SPLIT])
            print(f"Number of training examples: {len(train_ds)}")

            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
            val_dl = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            test_dl = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            # Use binary cross entropy loss
            criterion = nn.BCEWithLogitsLoss()

            ## No pretraining ##

            print("No pretraining")

            model = MODEL_CLASS(SEG_NUM_OUT_CHANNELS).to(DEVICE)
            optim = Adam(model.parameters(), lr=LR)

            train_loop(
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

            segmentation_image_output(
                model,
                test_dl,
                os.path.join(EXAMPLE_IMAGES_DIR, NO_PRETRAIN_SEG_NAME + f"_size_{len(train_ds)}_run_{run}" + ".jpg"),
                DEVICE,
            )

            model = model.to(dtype=torch.float32)
            torch.save(
                model.state_dict(),
                os.path.join(SAVED_MODEL_DIR, NO_PRETRAIN_SEG_NAME + f"_size_{len(train_ds)}_run_{run}" + ".pt"),
            )

            print("Done\n")

            ## Kaggle pretrain segmentation ##

            print("Image segmentation train with Kaggle dogs and cats pretrain")
            model = MODEL_CLASS(PRETRAIN_NUM_OUT_CHANNELS)
            model.load_state_dict(
                torch.load(os.path.join(SAVED_MODEL_DIR, KAGGLE_PRETRAIN_NAME + ".pt"), map_location=DEVICE)
            )
            model.new_head(SEG_NUM_OUT_CHANNELS)
            model = model.to(DEVICE)
            optim = Adam(model.parameters(), lr=LR)

            train_loop(
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

            segmentation_image_output(
                model,
                test_dl,
                os.path.join(EXAMPLE_IMAGES_DIR, KAGGLE_SEG_NAME + f"_size_{len(train_ds)}_run_{run}" + ".jpg"),
                DEVICE,
            )

            model = model.to(dtype=torch.float32)
            torch.save(
                model.state_dict(),
                os.path.join(SAVED_MODEL_DIR, KAGGLE_SEG_NAME + f"_size_{len(train_ds)}_run_{run}" + ".pt"),
            )

            print("Done\n")

            ## Synthetic pretrain segmentation ##

            print("Image segmentation train with synthetic data pretrain")
            model = MODEL_CLASS(PRETRAIN_NUM_OUT_CHANNELS)
            model.load_state_dict(
                torch.load(os.path.join(SAVED_MODEL_DIR, SYNTH_PRETRAIN_NAME + ".pt"), map_location=DEVICE)
            )
            model.new_head(SEG_NUM_OUT_CHANNELS)
            model = model.to(DEVICE)
            optim = Adam(model.parameters(), lr=LR)

            train_loop(
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

            segmentation_image_output(
                model,
                test_dl,
                os.path.join(EXAMPLE_IMAGES_DIR, SYNTH_SEG_NAME + f"_size_{len(train_ds)}_run_{run}" + ".jpg"),
                DEVICE,
            )

            model = model.to(dtype=torch.float32)
            torch.save(
                model.state_dict(),
                os.path.join(SAVED_MODEL_DIR, SYNTH_SEG_NAME + f"_size_{len(train_ds)}_run_{run}" + ".pt"),
            )

            print("Done\n")
