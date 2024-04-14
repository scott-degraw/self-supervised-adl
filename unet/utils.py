"""
Classes and functions used in training. Includes dataset classes, class for generating masks,
testing model, and visualizing model output.
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

from torch.cuda.amp import autocast 

from PIL import Image

from torchmetrics.classification import JaccardIndex


class CheckerboardMask(nn.Module):
    """
    Class to generate a mask for inpainting.
    """

    def __init__(self, square_size: int = 10, image_size: Tuple[int, int] = (240, 240)):
        """
        Args:
            square_size (int, optional): Size of square size used in the mask. Defaults to 10.
            image_size (Tuple[int, int], optional): Size of image to perform mask on. Defaults to (240, 240).
        """
        super().__init__()
        self.square_size = square_size
        self.image_size = image_size

        self.mask = torch.ones(image_size)
        self.mask.unsqueeze_(0)  # For broadcasting

        # Generate the mask
        h, w = self.mask.size(1), self.mask.size(2)
        for i in range(0, h, self.square_size):
            for j in range(0, w, self.square_size):
                if (i // self.square_size + j // self.square_size) % 2 == 0:
                    self.mask[:, i : i + self.square_size, j : j + self.square_size] = 0

    def __call__(self) -> torch.tensor:
        """
        Return a mask.

        Returns:
            torch.Tensor: Image mask.
        """
        return self.mask


"""
Datasets
"""


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


def images_mean_and_std(fnames: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Based of list of filenames of images calculate the per-channel
    mean and standard deviation.

    Args:
        fnames (list[str]): List of image filenames.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            1st item: Per channel means.
            2nd item: Per channel standard deviations.
    """

    channel_sums = torch.zeros((3,))
    square_channel_sums = torch.zeros((3,))

    num_pixels = 0

    for image_number, fname in enumerate(fnames):
        print(f"{image_number / len(fnames) * 100:4.1f} done \r", end="")
        image = read_image(fname)
        image = image[:3, :, :]  # If there is a 4th channel ignore it
        image = F.convert_image_dtype(image)
        num_pixels += image.numel() / image.shape[0]  # Don't count the subpixels
        summed_image = image.sum((1, 2))
        channel_sums += summed_image
        squared_summed_image = image.pow(2).sum((1, 2))
        square_channel_sums += squared_summed_image

    means = channel_sums / num_pixels
    std = torch.sqrt(square_channel_sums / num_pixels - means.pow(2))

    return means, std


class Trimap2Class(nn.Module):
    """
    Class to convert forground, background and unclassified labels to per pixel binary labels
    """

    def __init__(self):
        super().__init__()
        self.foreground = 1
        self.background = 2
        self.not_classified = 3

    def __call__(self, trimap: torch.Tensor) -> torch.Tensor:
        """
        Calculate the binary labels for the trimap

        Args:
            trimap (torch.Tensor): Trimap to calculate per pixel binary labels for.

        Returns:
            torch.Tensor: Image with with per pixel binary labels.
        """

        out: torch.Tensor = torch.ones_like(trimap, dtype=torch.float32)
        # Foreground is already correct
        out[trimap == self.background] = 0.0
        out[trimap == self.not_classified] = 0.5

        return out


class OxfordPetsDataset(Dataset):
    """
    PyTorch Dataset for the Oxford Pets Dataset. This is a wrapper
    over the PyTorch OxfordIIITPet dataset.

    Image: RGB images
    Trimap: 1:Foreground, 2:Background, 3:Not Classified

    We give the foreground pixels a value of 1.0, background 0.0, and not classified as 0.5.
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"] = "train",
        image_size: Tuple[int, int] = (240, 240),
    ):
        """
        Args:
            root (str): Root directory at which data will be downloaded.
            split (Literal[&quot;train&quot;, &quot;test&quot;], optional): Whether to use train or test split.
                Defaults to "train".
            image_size (Tuple[int, int], optional): The image size that the dataset images
                will be resized to. Defaults to (240, 240).
        """

        self.split = split
        # These values were precalculated
        image_means = torch.tensor((0.4778641164302826, 0.443441778421402, 0.3939882814884186))
        image_stds = torch.tensor((0.2677248418331146, 0.26299381256103516, 0.269730806350708))
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.Normalize(image_means, image_stds),
            ]
        )
        # Useful transformation to unnormalize images for displaying
        self.image_unnormalize = transforms.Compose(
            [
                transforms.Normalize(torch.zeros_like(image_means), 1 / image_stds),
                transforms.Normalize(-image_means, torch.ones_like(image_stds)),
            ]
        )

        self.trimap_transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                Trimap2Class(),
                transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.NEAREST),
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
            transform=self.image_transform,
            target_transform=self.trimap_transform,
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
            Tuple[torch.Tensor, torch.Tensor]:
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
        return F.to_pil_image(trimap)


class KaggleDogsAndCats(Dataset):
    """
    Dataset for kaggle dogs vs cats dataset. Downloads dataset if not already downloaded.

    Dataset consists of just images.
    """

    _dataset_name = "kaggle_dogs_and_cats"
    # Download url
    _url = "https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabstd_ucl_ac_uk/EXBk_s4yNOxPusy3f85vaHwBFYpd4uzW0dnHSTKjjt1j3A?e=uicsaz&download=1"
    _train_dir = "train"
    _test_dir = "test1"

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"] = "train",
        image_size: Tuple[float, float] = (240, 240),
    ):
        """
        Args:
            root (str): Root directory at which data will be downloaded.
            split (Literal[&quot;train&quot;, &quot;test&quot;], optional): Whether to use train or test split.
                Defaults to "train".
            image_size (Tuple[int, int], optional): The image size that the dataset images
                will be resized to. Defaults to (240, 240).
        """
        # These are precalculated
        image_means = torch.tensor((0.48621025681495667, 0.4532492756843567, 0.415396124124527))
        image_stds = torch.tensor((0.26264405250549316, 0.25600874423980713, 0.2586348354816437))
        self.image_transform = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(size=image_size, antialias=True),
                transforms.Normalize(image_means, image_stds),
            ]
        )
        self.image_unnormalize = transforms.Compose(
            [
                transforms.Normalize(torch.zeros_like(image_means), 1 / image_stds),
                transforms.Normalize(-image_means, torch.ones_like(image_stds)),
            ]
        )

        super().__init__()

        self.image_size = image_size

        download_dataset(self._url, root, self._dataset_name)

        # Get list of image filenames
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
        if self.image_transform is not None:
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
    """
    Dataset for stable diffusion synthetic dataset. Downloads dataset if not already downloaded.

    Dataset consists of just images.
    """

    _dataset_name = "stable_diffusion_images"
    _url = "https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabstd_ucl_ac_uk/EYwlyTA_OJRFjOdG_Pyghy4BzRN_oRluOIKsJia8u-YarQ?e=UsC46i&download=1"
    _train_dir = "ef_synthetic_pets"

    def __init__(self, root: str, image_size: tuple[int] = (240, 240)):
        """
        Args:
            root (str): Root directory at which data will be downloaded.
            split (Literal[&quot;train&quot;, &quot;test&quot;], optional): Whether to use train or test split.
                Defaults to "train".
            image_size (Tuple[int, int], optional): The image size that the dataset images
                will be resized to. Defaults to (240, 240).
        """

        super().__init__()
        self.image_size = image_size
        self._image_dir = os.path.join(root, self._dataset_name, self._train_dir)

        download_dataset(self._url, root, self._dataset_name)
        self.image_fnames = glob("*.png", root_dir=self._image_dir)

        image_means = torch.tensor((0.45749059319496155, 0.45294874906539917, 0.37832725048065186))
        image_stds = torch.tensor((0.25951141119003296, 0.25185921788215637, 0.26507002115249634))
        self.image_transform = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize(size=image_size, antialias=True),
                transforms.Normalize(image_means, image_stds),
            ]
        )
        self.image_unnormalize = transforms.Compose(
            [
                transforms.Normalize(torch.zeros_like(image_means), 1 / image_stds),
                transforms.Normalize(-image_means, torch.ones_like(image_stds)),
            ]
        )

    def __len__(self):
        """
        Returns:
            int: Number of images in dataset.
        """
        return len(self.image_fnames)

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        """
        Args:
            index (int): Index of image to retrieve.

        Returns:
            torch.Tensor: Image
        """
        image_fname = self.image_fnames[index]
        image = read_image(os.path.join(self._image_dir, image_fname))

        if self.image_transform is not None:
            image = self.image_transform(image)

        return image


class PretrainingDataset(Dataset):
    """
    Dataset wrapper to sample batch and masks during training.
    """

    def __init__(self, dataset: Dataset, mask_generator: nn.Module):
        """
        Args:
            dataset (Dataset): Dataset to wrap.
            mask_generator (nn.Module): The generator that will be used to generate the mask.
        """
        super().__init__()
        self.dataset = dataset
        self.image_transforms = dataset.image_transform
        self.image_unnormalize = dataset.image_unnormalize

        self.mask = mask_generator

    def __len__(self) -> int:
        """
        Returns:
            int: Number of images in dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index of image to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                1st item: Image.
                2nd item: Image mask.
        """

        return self.dataset[index], self.mask()


"""
Training/Testing 
"""


class InPaintingLoss(nn.Module):
    """
    Loss function for inpainting pretraining task. This loss only considers the reconstruction
    of the masked parts of the image not the unmasked.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, model: nn.Module, images: torch.Tensor, masks: torch.Tensor) -> float:
        """
        Calculate the inpainting loss for a model and a batch of images.

        Args:
            model (nn.Module): Model to evaluate loss on.
            images (torch.Tensor): Batch of images to evaluate loss on.
            masks (torch.Tensor): Batch of masks to evaluate loss on.

        Returns:
            float: Inpainting loss.
        """

        inverse_mask = 1 - masks

        norm = masks.count_nonzero()
        loss = (inverse_mask * (images - model(masks * images))).pow(2).sum() / norm

        return loss


def model_iou(model: nn.Module, eval_dl: DataLoader, device: torch.device) -> float:
    """
    Calculate the interesction of union (IOU) score for a model and a evaluation dataloader.

    Args:
        model (nn.Module): Model to evaluate IOU on.
        eval_dl (DataLoader): Dataloader to evaluate IOU.
        device (torch.device): Device that the model is on.

    Returns:
        float: IOU score.
    """

    model.eval()

    jaccard = JaccardIndex(task="binary", ignore_index=2).to(device)

    with torch.no_grad():
        for images, trimaps in eval_dl:
            images = images.to(device)
            trimaps = trimaps.to(device)

            with autocast():
                logit_pred = model(images)
                # Use the sign to make the foreground-background prediction.
                preds = (torch.sign(logit_pred) + 1) / 2

            # We want to ignore the "Not-classified" pixels of the trimap so set the 0.5 to 2
            # so we can ingore this class in the classifications
            trimaps[trimaps == 0.5] = 2
            jaccard.update(preds, trimaps)

    return jaccard.compute().item()


def segmentation_image_output(model: nn.Module, dl: DataLoader, fname: str, device: torch.device):
    """
    Visualise the results of the segmentation training.

    Args:
        model (nn.Module): Segmentation model.
        dl (DataLoader): Dataloader to produce example images from.
        fname (str): Filename to output results to.
        device (torch.device): Device that the model is on.
    """

    model.eval()
    with torch.no_grad():
        for images, targets in dl:
            images = images.to(device)
            targets = targets.to(device)

            pred_logits = model(images)
            dataset = dl.dataset
            # Get the image_unnormalize function from dataset and extract from various dataset wrappers
            while not hasattr(dataset, "image_unnormalize"):
                dataset = dataset.dataset
            images = dataset.image_unnormalize(images)
            # Make predictions based on sign of output and make sure dimensions are correct.
            pred = (torch.sign(pred_logits.expand(-1, images.shape[1], -1, -1)) + 1) / 2

            targets = targets.expand(-1, images.shape[1], -1, -1)
            save_image(torch.cat((images, targets, pred), dim=2), fname)
            break


def pretrain_image_output(model: nn.Module, dl: DataLoader, fname: str, device: torch.device):
    """
    Visualise the results of the pretraining.
    The parts of the image that are not inpainted are displayed with the original image pixels.

    Args:
        model (nn.Module): Pretrained model.
        dl (DataLoader): Dataloader to produce example images from.
        fname (str): Filename to output results to.
        device (torch.device): Device that the model is on.
    """

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
            while not hasattr(dataset, "image_unnormalize"):
                dataset = dataset.dataset

            masked_images = dataset.image_unnormalize(masked_images)
            outputs = dataset.image_unnormalize(outputs)
            # For the non masked parts output the true image.
            outputs = (1 - masks) * outputs + masks * masked_images

            # Make sure masked parts of images are white instead of grey from image normalisation
            masked_images += 1 - masks
            save_image(torch.cat((masked_images, outputs), dim=2), fname)
            break
