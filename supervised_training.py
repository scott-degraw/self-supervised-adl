import os
import torch
from torch.utils.data import DataLoader

from unet.model import UNet
from unet.datasets import OxfordPetsDataset
from unet.trainer import Trainer

if __name__ == "__main__":
    print("-- Supervised UNet Training on the Oxford Pets Dataset --")
    
    runs_dir = os.path.join(os.getcwd(),'runs')
    data_dir = os.path.join(os.getcwd(),'data','oxford_pets')
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    assert(os.path.exists(data_dir))
    
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    
    # Training Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 5

    # Load dataset(s) and dataloader(s)
    trainset = OxfordPetsDataset(data_dir, train=True, image_size=(240,240))
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Define the model
    model = UNet()
    
    # Define the trainer
    trainer = Trainer(
        model=model,
        optim=torch.optim.Adam(model.parameters(), lr=learning_rate),
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        checkpoint_name="unet_20240312T160038.pt",
    )
    
    # Train the model
    trainer.train(
        train_loader=trainloader,
        num_epochs=num_epochs,
        verbose=True,
        save_checkpoint=True,
    )