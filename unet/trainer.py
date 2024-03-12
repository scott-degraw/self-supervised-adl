import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    """General Trainer Class"""
    
    def __init__(
        self, model: nn.Module,
        optim: Optimizer,
        criterion: nn.Module,
        device: str | torch.DeviceObjType = torch.device('cpu'),
        checkpoint_name: str | None = None,
    ):
        self.model = model.to(device)
        self.optim = optim
        self.criterion = criterion
        self.device = device
        self.runs_dir = os.path.join(os.getcwd(),'runs')
        if checkpoint_name:
            self.load_checkpoint(checkpoint_name)
        
        print("Trainer initialized.")


    def load_checkpoint(self, checkpoint_name:str):
        model_file_path = os.path.join(os.getcwd(),'runs', checkpoint_name)
        self.model.load_state_dict(torch.load(model_file_path))
        print("Model checkpoint loaded.")
       
          
    def save_checkpoint(self):
        save_name = f"unet_{datetime.now().strftime('%Y%m%dT%H%M%S')}.pt"
        torch.save(self.model.state_dict(), os.path.join(self.runs_dir, save_name))
        print("Model checkpoint saved.")
        return save_name
        
        
    def _log(self, args):
        with open(os.path.join(self.runs_dir,'logs.txt'), 'at') as f:
            f.write("%s\n" % args)
    
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_epochs: int = 1,
        verbose: bool = True,
        save_checkpoint: bool = True
    ):
        print("Begin training model.")
        start_time = datetime.now()
        for epoch in range(num_epochs):
            train_epoch_loss = self._train(train_loader)
            
            val_epoch_loss = 0.0
            if val_loader is not None:
                val_epoch_loss = self._validate(val_loader)
            
            log = ("Epoch %i - Train loss: %.4f - Val loss: %.4f" % 
                    (epoch+1, train_epoch_loss, val_epoch_loss))
            self._log(log)
            if verbose:
                print(log)
                
        train_time = str(datetime.now()-start_time).split(".")[0]
        print("Model trained.")
        
        if save_checkpoint:
            save_name = self.save_checkpoint()
            log = ("Epochs: %i - Train time: %s - Train loss: %.4f - Val loss: %.4f - Filename: %s\n" % 
                    (num_epochs, train_time, train_epoch_loss, val_epoch_loss, save_name))
            self._log(log)
        
        
    def _train(self, train_loader: DataLoader):
        running_loss = 0.0
        steps = 0
        for images, trimaps in train_loader:
            self.optim.zero_grad()
            
            images = images.to(self.device)
            trimaps = trimaps.squeeze(1) - 1
            trimaps = trimaps.to(self.device)
            
            logits = self.model(images)
            loss = self.criterion(logits, trimaps)
            loss.backward()
            self.optim.step()
            
            running_loss += loss.item()
            steps += 1
        
        return running_loss / steps
    
    
    def _validate(self, val_loader: DataLoader):
        running_loss = 0.0
        steps = 0
        self.model.eval()
        for images, trimaps in val_loader:                
            images = images.to(self.device)
            trimaps = trimaps.squeeze(1)
            trimaps = trimaps.to(self.device)
            
            logits = self.model(images)
            loss = self.criterion(logits, trimaps)
            
            running_loss += loss.item()
            steps += 1
            self.model.train()
        
        return running_loss / steps