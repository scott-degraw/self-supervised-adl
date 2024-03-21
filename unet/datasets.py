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
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as F


def unzip_file(file_path: str):
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
    Download and extract dataset. 
    """
    dataset_save_path = os.path.join(root, dataset_name)
    zip_save_path = os.path.join(dataset_save_path, dataset_name + ".zip")

    # File that says whether data been extracted
    is_extracted_file = os.path.join(dataset_save_path, 'is_extracted')

    is_extracted = os.path.exists(is_extracted_file)

    if is_extracted:
        print(f"{dataset_name} already downloaded and extracted")
        return 
        
    elif not os.path.exists(zip_save_path):
        os.makedirs(dataset_save_path, exist_ok=True)
        print(f"{dataset_name} downloading")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        bytes_downloaded = 0

        with open(zip_save_path, 'wb') as f:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bytes_downloaded += len(data)
                progress = bytes_downloaded / total_size * 100 if total_size > 0 else 0
                print(f"Download progress: {progress:.2f}%\r", end='')
            print(f"{dataset_name} downloaded successfully.")

    print(f"Unzipping dataset")
    unzip_file(zip_save_path)
    
    # Create is_extracted file
    with open(os.path.join(dataset_save_path, 'is_extracted'), 'w') as _:
        pass

def trimap2image(trimap: torch.Tensor):
    """
    Create PIL image from trimap tensor.
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


class OxfordPetsDataset(Dataset):
    """
    PyTorch Dataset for the Oxford Pets Dataset
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"] = "train",
        image_size: Tuple[float, float] = (240, 240),
    ):
        image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
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

        trimap_transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x.type(torch.long)),
                transforms.Resize(size=image_size, antialias=True),
            ]
        )

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_image(self, index):
        """
        Returns PIL image at index.
        """
        image, _ = self.dataset[index]
        image = self.image_unnormalize(image)
        image = F.to_pil_image(image)
        return image

    def get_trimap(self, index):
        """
        Returns PIL trimap image 
        """
        _, trimap = self.dataset[index]
        trimap = trimap2image(trimap)
        return trimap


class KaggleDogsAndCats(Dataset):
    _dataset_name = "kaggle_dogs_vs_cats"
    _url = "https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabstd_ucl_ac_uk/EXBk_s4yNOxPusy3f85vaHwBFYpd4uzW0dnHSTKjjt1j3A?e=uicsaz&download=1"
    _train_dir = 'train'
    _test_dir = 'test1'

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

        download_dataset(self._url, root, self._dataset_name)

        self._data_dir = os.path.join(root, self._dataset_name)
        if split == "train":
            self._image_dir = os.path.join(self._data_dir, self._train_dir)
        else:
            self._image_dir = os.path.join(self._data_dir, self._test_dir)

        if not os.path.exists(self._image_dir):
            unzip_file(os.path.join(self._image_dir + '.zip'))

        self.image_fnames = glob("*.jpg", root_dir=self._image_dir)

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, index):
        image_fname = self.image_fnames[index]
        image = read_image(os.path.join(self._image_dir, image_fname))
        if self.image_transform:
            image = self.image_transform(image)
        
        return image

    def get_image(self, index):
        """
        Return PIL image at index.
        """
        image = self.__getitem__(index)
        image = self.image_unnormalize(image)
        image = F.to_pil_image(image)
        return image