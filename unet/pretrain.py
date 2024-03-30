import torch
import torch.nn as nn
from torch.utils.data import random_split

from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from os.path import join

from utils import *
from model import UNet

# Throw error if cuda not available (sorry mac people)
assert torch.cuda.is_available(), "CUDA not available"
DEVICE = torch.device("cuda")
SCALER = GradScaler()


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


if __name__ == "__main__":
    torch.manual_seed(438792)
    batch_size = 32
    eval_batch_size = 64
    max_num_epochs = 5
    patience = 5

    square_size = 16

    image_size = (240, 240)

    root_dir = "../data"
    ds = SynthDataset(root_dir, image_size=image_size)
    pretrain_ds = PretrainingDataset(ds, CheckerboardMask(square_size=square_size, image_size=ds.image_size))

    split_frac = 0.8
    train_ds, val_ds, test_ds = random_split(
        pretrain_ds, [split_frac * split_frac, split_frac * (1 - split_frac), 1 - split_frac]
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False)

    # Model
    model = UNet().to(DEVICE)

    # Loss
    criterion = InPaintingLoss()

    # Optimizer
    lr = 1e-3
    optim = Adam(model.parameters(), lr=lr)

    # Training Loop
    train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=max_num_epochs,
        patience=patience,
    )

    # Test model
    test_loss = test_step(test_dl, model, criterion)
    print(f"Test Loss: {test_loss:.4g}")

    # Test output
    pretrain_image_output(model, test_dl, "pretrain_output.png", DEVICE)

    # Save the model
    model = model.to(torch.device("cpu"), torch.float64)
    torch.save(model.state_dict(), "unet_pets.pth")

    # model.load_state_dict(torch.load('unet_pets.pth'))
    # model.to(DEVICE)

    # save_image_output(model, test_dl, 'test.png', DEVICE)

    print("Done")
