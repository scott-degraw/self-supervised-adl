from os import path 
from os.path import join

import PIL

import torch
import torchvision
import torch.nn as nn

from torch.optim import Adam 
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import random_split

from utils import * 
from model import UNet

# Throw error if cuda not available (sorry mac people)
assert torch.cuda.is_available(), "CUDA not available"
DEVICE = torch.device('cuda')
SCALER = GradScaler()


'''
Training/Testing Loops
''' 
def epoch_step(train_dl:torch.utils.data.DataLoader, model:nn.Module, criterion:nn.Module, optimizer:torch.optim.Optimizer) -> float:
    '''
    Do one epoch training Step
    Inputs:
        - train_dl (data.DataLoader): training dataloader
        - model (nn.Module): model to train
        - criterion (nn.Module): loss function
        - optimizer (optim.Optimizer): optimizer
    Returns:
        - total_loss: total loss for the epoch
    '''
    total_loss = 0.
    model.train()
    for (inputs, targets) in train_dl:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += inputs.shape[0] * loss.item()

        # Backward pass and Optimize
        SCALER.scale(loss).backward()
        SCALER.step(optimizer)
        SCALER.update()

    return total_loss / len(train_dl.dataset)


def test_step(test_dl:torch.utils.data.DataLoader, model:nn.Module, criterion:nn.Module) -> float:
    '''
    Test using the validation/test set
    Inputs:
        - test_dl (data.DataLoader): test dataloader
        - model (nn.Module): model to test
        - criterion (nn.Module): loss function
    Returns:
        Loss for the test set
    '''
    total_loss = 0.
    model.eval()
    with torch.no_grad():
        for (inputs, targets) in test_dl:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += targets.shape[0] * loss.item()

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
    val_scores = []
    best_val_score = -torch.inf
    no_improvement_counter = 0

    model_state_dicts = []

    for epoch in range(max_num_epochs):
        # Train
        epoch_loss = epoch_step(train_dl, model, criterion, optim)
        # Test
        val_score = model_iou(model, val_dl, DEVICE)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4g} Val iou: {val_score:.4g}")

        # Early stopping
        val_scores.append(val_score)

        cpu_state_dict = {key: tensor.cpu() for key, tensor in model.state_dict().items()}
        model_state_dicts.append(cpu_state_dict)

        if val_score > best_val_score:
            no_improvement_counter = 0
            best_val_score = val_score
        else:
            no_improvement_counter += 1

        if no_improvement_counter == patience:
            break

    best_epoch = torch.tensor(val_scores).argmax()
    best_val_score = val_scores[best_epoch]
    model.load_state_dict(model_state_dicts[best_epoch])

if __name__ == '__main__':
    torch.manual_seed(879)
    batch_size = 32
    eval_batch_size = 64
    max_num_epochs = 5
    patience = 5
    split = 0.8

    image_size = (240, 240)

    root_dir = "../data"
    train_val_ds = OxfordPetsDataset(root=root_dir, split="train", image_size=image_size)
    test_ds = OxfordPetsDataset(root=root_dir, split="test", image_size=image_size)

    train_ds, val_ds = random_split(train_val_ds, [split, 1 - split])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False)

    model = UNet(num_out_channels=1)
    model = model.to(DEVICE, torch.float32)

    # Loss
    criterion = nn.BCEWithLogitsLoss()

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

    # Save the model

    torch.save(model.state_dict(), 'unet_segmentation.pt')

    # Test model
    test_score = model_iou(model, test_dl, DEVICE)
    print(f'Test Loss: {test_score}')

    model.load_state_dict(torch.load('unet_segmentation.pt'))
    model.to(DEVICE)

    # Test output
    segmentation_image_output(model, test_dl, 'oxford_output.png', DEVICE)

    print('Done')