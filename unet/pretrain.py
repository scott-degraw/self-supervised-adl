import torch
import torch.nn as nn
from torch.utils.data import random_split

from torch.optim import Adam
from torch.cuda.amp import autocast

from utils import *
from run_config import *

"""
Training/Testing Loops
"""

def epoch_step(
    train_dl: torch.utils.data.DataLoader, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer
) -> float:
    """
    Do one epoch training Step
    Inputs:
        - train_dl (data.DataLoader): training dataloader
        - model (nn.Module): model to train
        - criterion (nn.Module): loss function
        - optimizer (optim.Optimizer): optimizer
    Returns:
        - total_loss: total loss for the epoch
    """
    total_loss = 0.0
    model.train()
    for images, masks in train_dl:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Forward pass
        optimizer.zero_grad()
        with autocast():
            loss = criterion(model=model, images=images, masks=masks)
            total_loss += images.shape[0] * loss.item()

        # Backward pass and Optimize
        SCALER.scale(loss).backward()
        SCALER.step(optimizer)
        SCALER.update()

    return total_loss / len(train_dl.dataset)


def test_step(test_dl: torch.utils.data.DataLoader, model: nn.Module, criterion: nn.Module) -> float:
    """
    Test using the validation/test set
    Inputs:
        - test_dl (data.DataLoader): test dataloader
        - model (nn.Module): model to test
        - criterion (nn.Module): loss function
    Returns:
        Loss for the test set
    """
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, masks in test_dl:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass
            with autocast():
                loss = criterion(model=model, images=images, masks=masks)
                total_loss += images.shape[0] * loss.item()

    return total_loss / len(test_dl.dataset)


def train_loop(
    train_dl: DataLoader,
    val_dl: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optim: torch.optim.Optimizer,
    max_num_epochs: int = 10,
    patience: int = 5,
):
    val_losses = []
    best_val_loss = torch.inf
    no_improvement_counter = 0

    model_state_dicts = []

    for epoch in range(max_num_epochs):
        # Train
        epoch_loss = epoch_step(train_dl, model, criterion, optim)
        # Test
        val_loss = test_step(val_dl, model, criterion)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4g} Val loss: {val_loss:.4g}")

        # Early stopping
        val_losses.append(val_loss)

        cpu_state_dict = {key: tensor.cpu() for key, tensor in model.state_dict().items()}
        model_state_dicts.append(cpu_state_dict)

        if val_loss < best_val_loss:
            no_improvement_counter = 0
            best_val_loss = val_loss
        else:
            no_improvement_counter += 1

        if no_improvement_counter == patience:
            break

    best_epoch = torch.tensor(val_losses).argmax()
    best_val_loss = val_losses[best_epoch]
    model.load_state_dict(model_state_dicts[best_epoch])

if __name__=="__main__":
    torch.manual_seed(7843718)
    
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

    train_loop(
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

    train_loop(
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