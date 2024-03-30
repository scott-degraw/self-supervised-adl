"""
Extends torch Dataset class to interface with the following datasets:
- Oxford Pets
- Stanford Dogs (TBD)
- Microsoft Dogs vs. Cats (TBD)
- Cat Head Detection (TBD)

"""

import os
import requests
import zipfile
from glob import glob
from typing import Literal
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as F

from PIL import Image


'''
Transforms
'''
class CheckerboardTransform(nn.Module):
    def __init__(self, square_size:int=10, image_size: Tuple[int, int] = (240, 240)):
        '''
        Make Checkerboard pattern on the image
        Args:
            - square_size (int): size of the squares, default 10
        '''
        super().__init__()
        self.square_size = square_size
        self.image_size = image_size

        self.mask = torch.ones(image_size)
        self.mask.unsqueeze_(0)

        h, w = self.mask.size(1), self.mask.size(2)
        for i in range(0, h, self.square_size):
            for j in range(0, w, self.square_size):
                if (i // self.square_size + j // self.square_size) % 2 == 0:
                    self.mask[:, i:i+self.square_size, j:j+self.square_size] = 0

    def __call__(self, image: torch.tensor) -> torch.tensor:
        return self.mask * image

class CheckerboardMask(nn.Module):
    def __init__(self, square_size:int=10, image_size: Tuple[int, int] = (240, 240)):
        '''
        Make Checkerboard pattern on the image
        Args:
            - square_size (int): size of the squares, default 10
        '''
        super().__init__()
        self.square_size = square_size
        self.image_size = image_size

        self.mask = torch.ones(image_size)
        self.mask.unsqueeze_(0)

        h, w = self.mask.size(1), self.mask.size(2)
        for i in range(0, h, self.square_size):
            for j in range(0, w, self.square_size):
                if (i // self.square_size + j // self.square_size) % 2 == 0:
                    self.mask[:, i:i+self.square_size, j:j+self.square_size] = 0

    def __call__(self) -> torch.tensor:
        return self.mask 

class RGBToBWTransform:
    def __call__(self, tensor:torch.tensor) -> torch.tensor:
        # Convert the normalized RGB tensor to a BW tensor
        bw_tensor = torch.mean(tensor, dim=0, keepdim=True)
        return bw_tensor
    

'''
Datasets
'''

def unzip_file(file_path: str):
    """
    Unzip file and extract into a directory with name of the zip file with the .zip extension removed.

    Args:
        file_path (str): File path to unzip.

    Raises:
        RuntimeError
    """
    assert os.path.exists(file_path), f"File path: {file_path} does not exist"

    extract_dir = os.path.dirname(file_path)
    _, ext = os.path.splitext(file_path)

    os.makedirs(extract_dir, exist_ok=True)

    if ext != ".zip":
        raise RuntimeError("File extension is not a zip file")

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def download_dataset(url: str, root: str, dataset_name: str):
    """
    Download and extract dataset. Checks to see if dataset has already been extracted
    and/or downloaded.

    Args:
        url (str): Url to download from.
        root (str): Directory to download dataset to.
        dataset_name (str): Name of dataset. This will be used to name the downloaded dataset directories.
    """
    dataset_save_path = os.path.join(root, dataset_name)
    zip_save_path = os.path.join(dataset_save_path, dataset_name + ".zip")

    # File that says whether data been extracted
    is_extracted_file = os.path.join(dataset_save_path, "is_extracted")

    is_extracted = os.path.exists(is_extracted_file)

    if is_extracted:
        print(f"{dataset_name} already downloaded and extracted")
        return

    elif not os.path.exists(zip_save_path):
        os.makedirs(dataset_save_path, exist_ok=True)
        print(f"{dataset_name} downloading")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        bytes_downloaded = 0

        with open(zip_save_path, "wb") as f:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bytes_downloaded += len(data)
                progress = bytes_downloaded / total_size * 100 if total_size > 0 else 0
                print(f"Download progress: {progress:.2f}%\r", end="")
            print(f"{dataset_name} downloaded successfully.")

    print(f"Unzipping dataset")
    unzip_file(zip_save_path)

    # Create empty is_extracted file
    with open(os.path.join(dataset_save_path, "is_extracted"), "w") as _:
        pass


def trimap2pil(trimap: torch.Tensor) -> Image:
    """
    Create PIL image from trimap tensor. This converts the border, background and pet class ids
    to colors that look good when displaying.
    Args:
        trimap (torch.Tensor): Trimap to extract image from. Shape (B, 1, H, W).
    """
    original_background = 2
    original_border = 3
    original_pet = 1

    background = 240
    border = 0
    pet = 128

    trimap[trimap == original_background] = background
    trimap[trimap == original_border] = border
    trimap[trimap == original_pet] = pet

    return F.to_pil_image(trimap.type(torch.uint8))

def images_mean_and_std(fnames: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    channel_sums = torch.zeros((3, ))
    square_channel_sums = torch.zeros((3, ))

    num_pixels = 0

    for image_number, fname in enumerate(fnames):
        print(f"{image_number / len(fnames) * 100:4.1f} done \r")
        image = read_image(fname)
        image = F.convert_image_dtype(image)
        num_pixels += image.numel() / image.shape[0] # Don't count the subpixels
        summed_image = image.sum((1, 2))
        channel_sums += summed_image       
        squared_summed_image = image.pow(2).sum((1, 2))
        square_channel_sums += squared_summed_image
    
    means = channel_sums / num_pixels
    std = torch.sqrt(square_channel_sums / num_pixels - means.pow(2))

    return means, std

class OxfordPetsDataset(Dataset):
    """
    PyTorch Dataset for the Oxford Pets Dataset

    Image: RGB images
    Trimap: 1:Foreground, 2:Background, 3:Not Classified
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"] = "train",
        image_size: Tuple[int, int] = (240, 240),
    ):
        """
        Initialise class.

        Args:
            root (str): Root directory at which data will be downloaded.
            split (Literal[&quot;train&quot;, &quot;test&quot;], optional): Whether to use train or test split.
                Defaults to "train".
            image_size (Tuple[int, int], optional): The image size that the dataset images
                will be resized to. Defaults to (240, 240).
        """
        image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=image_size, antialias=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # Useful transformation to unnormalize images for displaying
        self.image_unnormalize = transforms.Compose(
            [
                transforms.Normalize((0, 0, 0), (2.0, 2.0, 2.0)),
                transforms.Normalize((-0.5, -0.5, -0.5), (1.0, 1.0, 1.0)),
            ]
        )

        trimap_transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x.type(torch.long)),
                transforms.Resize(size=image_size, antialias=True),
            ]
        )

        super().__init__()

        self.image_size = image_size

        if split == "train":
            dataset_split = "trainval"
        else:
            dataset_split = "test"

        self.dataset = OxfordIIITPet(
            root=root,
            split=dataset_split,
            target_types="segmentation",
            download=True,
            transform=image_transform,
            target_transform=trimap_transform,
        )

    def __len__(self) -> int:
        """
        Return number of elements in dataset.

        Returns:
            int: Length of dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and segmentation map at specified index

        Args:
            index (int): Index at which to extract data point.

        Returns:
            tuple:
                1st item (torch.Tensor): Image. Shape (C, H, W).
                2nd item (torch.Tensor): Segmentation map. Shape (1, H, W).
        """
        return self.dataset[index]

    def get_image(self, index: int) -> Image:
        """
        Returns PIL image at index.

        Args:
            index (int): Index at which to extract a PIL image.
        """
        image, _ = self.dataset[index]
        image = self.image_unnormalize(image)
        image = F.to_pil_image(image)
        return image

    def get_trimap(self, index: int) -> Image:
        """
        Returns PIL trimap image.

        Args:
            index (int): Index at which to extract a PIL version of trimap.
        """
        _, trimap = self.dataset[index]
        trimap = trimap2pil(trimap)
        return trimap


class KaggleDogsAndCats(Dataset):
    """
    Dataset for kaggle dogs vs cats dataset.
    Downloads dataset if not present. 

    Datset consists of just images.
    """
    _dataset_name = "Kaggle dogs and cats"
    _url = "https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabstd_ucl_ac_uk/EXBk_s4yNOxPusy3f85vaHwBFYpd4uzW0dnHSTKjjt1j3A?e=uicsaz&download=1"
    _train_dir = "train"
    _test_dir = "test1"

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"] = "train",
        image_size: Tuple[float, float] = (240, 240),
    ):
        self.image_transform = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(size=image_size, antialias=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.image_unnormalize = transforms.Compose(
            [
                transforms.Normalize((0, 0, 0), (2.0, 2.0, 2.0)),
                transforms.Normalize((-0.5, -0.5, -0.5), (1.0, 1.0, 1.0)),
            ]
        )

        super().__init__()

        self.image_size = image_size

        download_dataset(self._url, root, self._dataset_name)

        self._data_dir = os.path.join(root, self._dataset_name)
        if split == "train":
            self._image_dir = os.path.join(self._data_dir, self._train_dir)
        else:
            self._image_dir = os.path.join(self._data_dir, self._test_dir)

        if not os.path.exists(self._image_dir):
            unzip_file(os.path.join(self._image_dir + ".zip"))

        self.image_fnames = glob("*.jpg", root_dir=self._image_dir)

    def __len__(self) -> int:
        """
        Returns:
            int: Number of images in dataset. 
        """
        return len(self.image_fnames)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Args:
            index (int): Index of image to retrieve. 

        Returns:
            torch.Tensor: Image
        """
        image_fname = self.image_fnames[index]
        image = read_image(os.path.join(self._image_dir, image_fname))
        if self.image_transform:
            image = self.image_transform(image)

        return image

    def get_image(self, index: int) -> Image:
        """
        Return PIL image at index.

        Args:
            index (int): Index at which to get PIL image.
        
        Returns:
            Image: PIL image.
        
        """
        image = self.__getitem__(index)
        image = self.image_unnormalize(image)
        image = F.to_pil_image(image)
        return image

class SynthDataset(Dataset):
    _dataset_name = "stable_diffusion_images"
    _url = "https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabstd_ucl_ac_uk/EZU5VGweR2JAv0i4Vfr-uvIBw8QkPT_7TTVQXQ4wlAs54w?e=kBDsnU&download=1"
    _train_dir = "stable_diffusion_images"

    def __init__(self, root: str, image_size: tuple[int] = (240, 240)):
        """
        Synthetic dataset
        Args:
            - root (str): root directory of the dataset
            - img_size (tuple[int]): image size to reshape to 
        """

        super().__init__()
        self.image_size = image_size
        self._image_dir = os.path.join(root, self._dataset_name, self._train_dir)

        download_dataset(self._url, root, self._dataset_name)
        self.image_fnames = glob("*.png", root_dir=self._image_dir)

        self.image_transform = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(size=image_size, antialias=True),
                transforms.Normalize((0.5498123168945312, 0.4941849112510681, 0.4348284602165222), 
                                     (0.278773695230484, 0.26160579919815063, 0.27657902240753174)
                                     ),
            ]
        )
        self.image_unnormalize = transforms.Compose(
            [
                transforms.Normalize((0, 0, 0), 
                                     (1 / 0.278773695230484, 1 / 0.26160579919815063, 1 / 0.27657902240753174)),
                transforms.Normalize((-0.5498123168945312, -0.4941849112510681, -0.4348284602165222),
                                     (1.0, 1.0, 1.0)),
            ]
        )


    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        """
        Get image and its corresponding BW image
        """
        image_fname = self.image_fnames[index]
        image = read_image(os.path.join(self._image_dir, image_fname))

        if self.image_transform is not None:
            image = self.image_transform(image)
        
        return image

class PretrainingDataset(Dataset):
    """
    Dataset to give image and mask for training
    """

    def __init__(self, dataset: Dataset, mask_generator: nn.Module):
        super().__init__()
        self.dataset = dataset
        self.image_transforms = dataset.image_transform
        self.image_unnormalize = dataset.image_unnormalize

        self.mask = mask_generator
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[index], self.mask()

        

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

# class InPaintingLoss(nn.Module):
#     def __init__(self, reco_weight:float = 0.99, context_weight:float = 0.01):
#         super().__init__()
#         self._reco_weight = reco_weight
#         self._context_weight = context_weight
    
#     def __call__(self, model: nn.Module, images: torch.Tensor, masks: torch.Tensor):
#         inverse_mask = 1 - masks
        
#         reco_norm = masks.count_nonzero()
#         reco_loss = (inverse_mask * (images - model(masks * images))).pow(2).sum() / reco_norm

#         context_norm = inverse_mask.count_nonzero()
#         context_loss = (masks * (images - model(inverse_mask * images))).pow(2).sum() / context_norm

#         loss = self._reco_weight * reco_loss + self._context_weight * context_loss

#         return loss

class InPaintingLoss(nn.Module):
    """
    Loss function for inpainting pretraining task. This loss only considers the reconstruction 
    of the masked parts of the image not the unmasked. 
    """
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, model: nn.Module, images: torch.Tensor, masks: torch.Tensor):
        inverse_mask = 1 - masks
        
        reco_norm = masks.count_nonzero()
        reco_loss = (inverse_mask * (images - model(masks * images))).pow(2).sum() / reco_norm

        return reco_loss

def iou_score(y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
    '''
    Compute Intersection over Union (IOU) score between predicted and target masks for binary classification.
    Ignores the non-classified 0.5 labels of the trimaps ie IOU metric will not take them into account.
    
    Parameters:
        y_pred (torch.tensor): Predicted masks with shape (batch_size, channels, height, width)
        y_true (torch.tensor): Target masks with shape (batch_size, channels, height, width)
        
    Returns:
        torch.tensor: IOU score (batch_size, )
    ''' 
    # Flatten each image
    y_pred = y_pred.reshape((y_pred.shape[0], -1))
    y_true = y_true.reshape((y_true.shape[0], -1))

    positives = y_pred == 1
    negatives = y_pred == 0
    
    true_positives = y_true == 1
    true_negatives = y_true == 0

    true_positives = torch.count_nonzero(positives.logical_and(true_positives))
    false_positives = torch.count_nonzero(positives.logical_and(true_negatives))
    false_negatives = torch.count_nonzero(negatives.logical_and(true_positives))

    # Add epsilon so that we do not divide by zero
    iou = (true_positives + 1e-9) / (true_positives + false_positives + false_negatives + 1e-9)

    return iou

def model_iou(model: nn.Module, eval_dl: DataLoader, device: torch.device):
    """
    Calculate the interesction of union (IOU) score for a model and a evaluation dataloader.
    """
    iou_sum = 0

    model.eval()
    with torch.no_grad():
        for images, trimaps in eval_dl:
            images = images.to(device)
            trimaps = trimaps.to(device)

            with autocast():
                logit_pred = model(images)
                pred = (torch.sign(logit_pred) + 1) / 2

            iou_sum += iou_score(pred, trimaps).sum()
    
    return iou_sum / len(eval_dl.dataset)

def segmentation_image_output(model: nn.Module, dl: DataLoader, fname: str, device: torch.device):
    """
    Visualise the results of the segmentation training.
    """

    model.eval()
    with torch.no_grad():
        for images, targets in dl:
            images = images.to(device)
            targets = targets.to(device)

            pred_logits = model(images)
            images = dl.dataset.image_unnormalize(images)
            pred_logits = nn.functional.sigmoid(pred_logits.expand(-1, images.shape[1], -1, -1))
            save_image(torch.cat((images, pred_logits), dim=2), fname)
            break


def pretrain_image_output(model: nn.Module, dl: DataLoader, fname: str, device: torch.device):
    '''
    Pull an image from the dataloader and save the network's prediction for pretraining.
    The parts of the image that are not inpainted are displayed with the original image part.
    '''

    model.eval()
    with torch.no_grad():
        for images, masks in dl:
            images = images.to(device)
            masks = masks.to(device)

            masked_images = masks * images

            # Forward pass
            outputs = model(masked_images)

            # Get the image_unnormalize function from dataset and extract from various dataset wrappers
            dataset = dl.dataset
            while hasattr(dataset, "dataset"):
                dataset = dataset.dataset

            masked_images = dataset.image_unnormalize(masked_images)
            outputs = dataset.image_unnormalize(outputs)
            # For the non masked parts output the true image.
            outputs = (1 - masks) * outputs + masks * masked_images 
            save_image(torch.cat((masked_images, outputs), dim=2), fname)
            break
