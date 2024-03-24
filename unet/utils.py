"""
Extends torch Dataset class to interface with the following datasets:
- Oxford Pets
- Stanford Dogs (TBD)
- Microsoft Dogs vs. Cats (TBD)
- Cat Head Detection (TBD)

"""

import os
#import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image


'''
Transforms
'''
class CheckerboardTransform:
    def __init__(self, square_size:int=10):
        '''
        Make Checkerboard pattern on the image
        Args:
            - square_size (int): size of the squares, default 10
        '''
        self.square_size = square_size

    def __call__(self, tensor:torch.tensor) -> torch.tensor:
        h, w = tensor.size(1), tensor.size(2)
        for i in range(0, h, self.square_size):
            for j in range(0, w, self.square_size):
                if (i // self.square_size + j // self.square_size) % 2 == 0:
                    tensor[:, i:i+self.square_size, j:j+self.square_size] = 0
        return tensor


class RGBToBWTransform:
    def __call__(self, tensor:torch.tensor) -> torch.tensor:
        # Convert the normalized RGB tensor to a BW tensor
        bw_tensor = torch.mean(tensor, dim=0, keepdim=True)
        return bw_tensor
    

'''
Datasets
'''
class OxfordPetsDataset(Dataset):
    """PyTorch Dataset for the Oxford Pets Dataset
    
    Code adapted from: https://github.com/PSRahul/semanticseg/blob/master/pythonfiles/data_class.py
    
    Image: RGB images
    Trimap: 1:Foreground, 2:Background, 3:Not Classified
    """
    def __init__(self, root:str, image_size:tuple[int]=(240,240)):
        '''
        Dataset for Oxford Pets
        Args:
            - root (str): root directory of the dataset
            - image_size (tuple[int]): size of the image to be returned, default (240,240)
        '''
        # Get all image names (images, annotations)
        labels = [os.path.splitext(os.path.basename(l))[0] for l in os.listdir(os.path.join(root, 'images'))]

        self.file_path_images = [os.path.join(root, 'images', l+".jpg") for l in labels]
        self.file_path_trimaps = [os.path.join(root, 'annotations', l+".png") for l in labels]
        
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=image_size, antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
            # transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)), # ImageNet mean & std
        ])
        self.trimap_transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize(size=image_size, antialias=True),
        ])
        
        #print(f"Loaded {len(self.file_path_images)} Images and Trimaps")
        return
        
    def __len__(self):
        return len(self.file_path_images)
    
    def __getitem__(self, index:int) -> tuple[torch.tensor, torch.tensor]:
        '''
        Get image and its corresponding trimap
        Inputs:
            - index: index of the image-trimap pair
        Returns:
            Tuple of image and trimap as pytorch tensors
        '''
        image_path = self.file_path_images[index]
        trimap_path = self.file_path_trimaps[index]
        
        image = Image.open(image_path).convert('RGB')
        trimap = Image.open(trimap_path)
        
        return self.image_transform(image), self.trimap_transform(trimap)
        # return {'image': self.image_transform(image), 'trimap': self.trimap_transform(trimap)}


class SynthDataset(Dataset):
    def __init__(self, root:str, img_size:tuple[int]=(240,240)):
        super(SynthDataset, self).__init__()
        '''
        Synthetic dataset
        Args:
            - root (str): root directory of the dataset
            - img_size (tuple[int]): image size to reshape to 
        '''
        self.image_list = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.png')] 

        self.x_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=img_size, antialias=True),
            CheckerboardTransform(square_size=16),
            #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
        ]) 
        self.y_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=img_size, antialias=True),
            #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
            #RGBToBWTransform(),
        ]) 

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index:int) -> tuple[torch.tensor, torch.tensor]:
        '''
        Get image and its corresponding BW image 
        '''

        image = Image.open(self.image_list[index]).convert('RGB')
        x_img = self.x_transform(image)
        y_img = self.y_transform(image)
        return x_img, y_img


def get_splits(ds:Dataset, batch_size:int=64, split:float=.8) -> tuple[Dataset, Dataset, Dataset]:
    '''
    Split the dataset into train, validation, and test sets
    Inputs:
        - ds: dataset to split
        - batch_size: batch size for the dataloaders
        - split (float): proportion of the dataset to use for training/val and testing (default=.8)
    Returns:
        - train_dl: training 
        - val_dl: validation 
        - test_dl: test 
    '''
    # Random shuffle the indices
    #indices = np.arange(len(ds))
    #np.random.shuffle(indices)
    indices = torch.randperm(len(ds))

    # Split the indices
    vt_split = indices[:int(split*len(ds))] #Need to train and test split
    train_split = vt_split[:int((split)*len(vt_split))]
    val_split = vt_split[int((split)*len(vt_split)):] 
    test_split = indices[int((split)*len(ds)):]

    # Subset: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset
    train_ds = Subset(ds, train_split)
    val_ds = Subset(ds, val_split)
    test_ds = Subset(ds, test_split)

    # Dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return (train_dl, val_dl, test_dl)


'''
Training/Testing 
'''
def iou_loss(y_pred:torch.tensor, y_true:torch.tensor) -> torch.tensor:
    '''
    Compute Intersection over Union (IoU) loss between predicted and target masks.
    
    Parameters:
        y_pred (torch.tensor): Predicted masks with shape (batch_size, channels, height, width)
        y_true (torch.tensor): Target masks with shape (batch_size, channels, height, width)
        
    Returns:
        torch.tensor: IOU loss (should be single value)
    ''' 
    intersection = torch.sum(y_pred * y_true, dim=(1, 2, 3))
    union = torch.sum(y_pred + y_true, dim=(1, 2, 3)) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)  # Adding epsilon to avoid division by zero
    
    return 1 - torch.mean(iou)


def save_image_output(network:nn.Module, dl:DataLoader, fname:str, device:torch.device):
    '''
    Pull an image from the dataloader and save the network's prediction
    '''

    network.eval()
    with torch.no_grad():
        for (inputs, targets) in dl:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = network(inputs)

            # Save the first image
            save_image(torch.cat((inputs, outputs), dim=2), fname)
            break
