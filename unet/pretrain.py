
import torch
import torch.nn as nn
from torch.optim import Adam 

from utils import * 
from model import UNet

'''
Device and Data Type
-> if cuda is available, use it
-> if not, use cpu
-> if mac, use mps (metal shaders)
'''
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
DTYPE = torch.float32

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
        inputs = inputs.to(DEVICE, dtype=DTYPE)
        targets = targets.to(DEVICE, dtype=DTYPE)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # Backward pass and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            inputs = inputs.to(DEVICE, dtype=DTYPE)
            targets = targets.to(DEVICE, dtype=DTYPE)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss


if __name__ == '__main__':
    folder = '../../adl_data/synthetic_data'

    # Load the dataset
    all_ds = SynthDataset(folder)
    train_dl, val_dl, test_dl = get_splits(all_ds, batch_size=16, split=.8)

    print(f'Lenght of train_dl: {len(train_dl)}')

    '''
    # Quick test
    for (inputs, targets) in train_dl:
        inputs = inputs.to(DEVICE, dtype=DTYPE)
        targets = targets.to(DEVICE, dtype=DTYPE)
        print(inputs.shape, targets.shape)
        break
    '''
    # Model
    network = UNet().to(DEVICE, dtype=DTYPE) 

    # Loss
    loss = nn.MSELoss()

    # Optimizer
    lr = 1e-3
    optim = Adam(network.parameters(), lr=lr)

    # Training Loop
    num_epochs = 10
    for e in range(num_epochs):
        # Train
        epoch_loss = epoch_step(train_dl, network, loss, optim)
        print(f'Epoch {e+1} Loss: {epoch_loss}')
        # Test
        t_loss = test_step(val_dl, network, loss)
        print(f'Epoch {e+1} Val Loss: {t_loss}')

    # Save the model
    torch.save(network.state_dict(), 'unet_pets.pth')

    # Test model
    t_loss = test_step(test_dl, network, loss)
    print(f'Epoch {e+1} Test Loss: {t_loss}')

    # Test output
    save_image_output(network, test_dl, 'test_output.png', DEVICE)

    print('Done')