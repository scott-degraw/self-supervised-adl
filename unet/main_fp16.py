import torch
import torch.nn as nn

from torch.optim import Adam 
from torch.cuda.amp import GradScaler, autocast
from os.path import join

from utils import * 
from unet_model import UNet

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
        targets = targets.squeeze(1) - 1

        # Forward pass
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

        # Backward pass and Optimize
        SCALER.scale(loss).backward()
        SCALER.step(optimizer)
        SCALER.update()

    return total_loss


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
            targets = targets.squeeze(1) - 1

            # Forward pass
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
    return total_loss



if __name__ == '__main__':
    data_dir = '/home/squirt/Documents/data'
    folder = join(data_dir, 'adl_data/oxford')

    # Load the dataset
    all_ds = OxfordPetsDataset(folder)
    train_dl, val_dl, test_dl = get_splits(all_ds, batch_size=32, split=.5)

    network = UNet().to(DEVICE) 
    network.load_state_dict(torch.load('unet_pets.pth'))

    # Loss
    loss = nn.CrossEntropyLoss()

    # Optimizer
    lr = 1e-3
    optim = Adam(network.parameters(), lr=lr)

    # Training Loop
    num_epochs = 3 
    for e in range(num_epochs):
        # Train
        epoch_loss = epoch_step(train_dl, network, loss, optim)
        print(f'Epoch {e+1} Loss: {epoch_loss}')
        # Test
        t_loss = test_step(val_dl, network, loss)
        print(f'Epoch {e+1} Val Loss: {t_loss}')

    # Save the model
    torch.save(network.state_dict(), 'unet_oxford.pth')

    # Test model
    t_loss = test_step(test_dl, network, loss)
    print(f'Epoch {e+1} Test Loss: {t_loss}')

    # Test output
    save_image_output(network, test_dl, 'test_output.png', DEVICE)

    print('Done')