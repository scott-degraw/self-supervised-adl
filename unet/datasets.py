"""
Extends torch Dataset class to interface with the following datasets:
- Oxford Pets
- Stanford Dogs (TBD)
- Microsoft Dogs vs. Cats (TBD)
- Cat Head Detection (TBD)

"""

import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class OxfordPetsDataset(Dataset):
    """PyTorch Dataset for the Oxford Pets Dataset
    
    Code adapted from: https://github.com/PSRahul/semanticseg/blob/master/pythonfiles/data_class.py
    
    Image: RGB images
    Trimap: 1:Foreground, 2:Background, 3:Not Classified
    """
    def __init__(self, root, train=False, val=False, test=False, image_size=(240,240)):
        if(train):
            label_filename = "train.txt"
        elif(val):
            label_filename = "val.txt"
        elif(test):
            label_filename = "test.txt"
        else:
            raise Exception("Select train or test set")
            
        with open(os.path.join(root,'annotations','splits',label_filename), 'rt') as f:
            labels = f.readlines() # reads labels formatted as <Image CLASS-ID SPECIES BREED-ID>
        labels = [l.split(' ')[0] for l in labels]
        
        self.images_dir = os.path.join(root,'images')
        self.trimaps_dir = os.path.join(root,'annotations','trimaps')
        
        self.file_path_images = [os.path.join(self.images_dir, l+".jpg") for l in labels]
        self.file_path_trimaps = [os.path.join(self.trimaps_dir, l+".png") for l in labels]
        
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
    
    def __getitem__(self, index):
        image_path = self.file_path_images[index]
        trimap_path = self.file_path_trimaps[index]
        
        image = Image.open(image_path).convert('RGB')
        trimap = Image.open(trimap_path)
        
        return self.image_transform(image), self.trimap_transform(trimap)
        # return {'image': self.image_transform(image), 'trimap': self.trimap_transform(trimap)}
    
    def show_image(self, index):
        image_path = self.file_path_images[index]
        
        image = np.array(Image.open(image_path).convert('RGB'))
        image = Image.fromarray(image)
        image = image.convert('RGB')
        image.show()
        
    def show_trimap(self, index):
        trimap_path = self.file_path_trimaps[index]
        
        image = np.array(Image.open(trimap_path))*80.0
        image = Image.fromarray(image)
        image = image.convert('L')
        image.show()
        
        
