"""
Extends torch Dataset class to interface with the following datasets:
- Oxford Pets
- Stanford Dogs (TBD)
- Microsoft Dogs vs. Cats (TBD)
- Cat Head Detection (TBD)

"""

import os
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
    _dataset_name = "kaggle_dogs_vs_cats"
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
    def __init__(self, root: str, img_size: tuple[int] = (240, 240)):
        super(SynthDataset, self).__init__()
        """
        Synthetic dataset
        Args:
            - root (str): root directory of the dataset
            - img_size (tuple[int]): image size to reshape to 
        """
        self.image_list = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".png")]

        self.x_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=img_size, antialias=True),
                CheckerboardTransform(square_size=16),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.y_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=img_size, antialias=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # RGBToBWTransform(),
            ]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        """
        Get image and its corresponding BW image
        """

        image = Image.open(self.image_list[index]).convert("RGB")
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
