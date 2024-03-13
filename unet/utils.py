"""
Extends torch Dataset class to interface with the following datasets:
- Oxford Pets
- Stanford Dogs (TBD)
- Microsoft Dogs vs. Cats (TBD)
- Cat Head Detection (TBD)

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

'''
Datasets
'''
class OxfordPetsDataset(Dataset):
    """PyTorch Dataset for the Oxford Pets Dataset
    
    Code adapted from: https://github.com/PSRahul/semanticseg/blob/master/pythonfiles/data_class.py
    
    Image: RGB images
    Trimap: 1:Foreground, 2:Background, 3:Not Classified
    """
    def __init__(self, root:str, image_size=(240,240)):
        '''
        Dataset for Oxford Pets
        Args:
            - root: root directory of the dataset
            - image_size: size of the image to be returned
        '''
        # Get all image names (images, annotations)
        labels = [os.path.splitext(os.path.basename(l))[0] for l in os.listdir(os.path.join(root, 'images'))]

        self.file_path_images = [os.path.join(self.images_dir, 'images', l+".jpg") for l in labels]
        self.file_path_trimaps = [os.path.join(self.trimaps_dir, 'annotations', l+".png") for l in labels]
        
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
        
        print(f"Loaded {len(self.file_path_images)} Images and Trimaps")
        
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
    indices = np.arange(len(ds))
    np.random.shuffle(indices)

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
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return (train_dl, val_dl, test_dl)
    