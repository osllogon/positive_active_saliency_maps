# deep learning libraries
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# other libraries
import os
import random
from typing import Tuple

# other libraries
from PIL import Image


def load_cifar10_data(path: str, batch_size: int = 128, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """_summary_

    Parameters
    ----------
    path : str
        _description_
    color_space : str
        _description_
    batch_size : int, optional
        _description_, by default 128
    num_workers : int, optional
        _description_, by default 0

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        _description_
    """
    
    transformations = transforms.Compose([
            transforms.ToTensor()
        ])
    train_dataset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transformations)
    test_dataset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transformations)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader, test_dataloader

class ImagenetteDataset(Dataset):

    def __init__(self, path: str) -> None:
        """
        Constructor of ImagenetteDataset
        
        Parameters
        ----------
        path : str
            path of the dataset
        color_space : str
            color space for loading the images
        
        Raises
        ------
        FileNotFoundError
            if the path of the dataset does not exist
        """

        self.path = path
        self.names = os.listdir(path)

    def __len__(self) -> int:
        """
        This method returns the length of the dataset
        
        Returns
        -------
        int
            length of dataset
        """

        return len(self.names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        This method loads an item based on the index
        
        Parameters
        ----------
        index : int
            index of the element in the dataset
            
        Returns
        -------
        torch.Tensor
            image. Dimensions: [channels, height, width]
        int
            label
        """

        # load image path and label
        image_path = f'{self.path}/{self.names[index]}'
        label = int(self.names[index].split('_')[0])

        # load image 
        transformations = transforms.Compose([
            transforms.ToTensor()
        ])
        image = Image.open(image_path)
        image = transformations(image)

        return image, label



def load_imagenette_data(path: str, batch_size: int = 128, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    This function returns two Dataloaders, one for train data and other for validation data for imagenette dataset
    
    Parameters
    ----------
    path : str
        path of the dataset
    color_space : str
        color_space for loading the images
    batch_size : int, Optional
        batch size for dataloaders. Default value: 128
    num_workers : int
        number of workers for loading data. Default value: 0
        
    Returns
    -------
    DataLoader
        train data
    DataLoader
        validation data
    """

    # create datasets
    train_datatset = ImagenetteDataset(f'{path}/train')
    val_dataset = ImagenetteDataset(f'{path}/val')

    # define dataloaders
    train_dataloader = DataLoader(train_datatset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader, val_dataloader


def preprocess_imagenette(original_path: str, save_path: str) -> None:
    """
    This method generates a single directory for each split (train and val) with all the images and the title of each
    image is in the following format: {class_number}_{random_number}.jpg. This method also resizes all images to 256x256
    
    Parameters
    ----------
    original_path : str
        original path for the dataset
    save_path : str
        path for saving the dataset after preprocess
        
    Raises
    ------
    FileNotFoundError
        if original_path does not exist
    """

    # create save directories if they don't exist
    if not os.path.isdir(f'{save_path}'):
        os.makedirs(f'{save_path}/train')
        os.makedirs(f'{save_path}/val')

    # define resize transformation
    transform = transforms.Resize((224, 224))

    # loop for saving processed data
    list_splits = os.listdir(original_path)
    for i in range(len(list_splits)):
        list_class_dirs = os.listdir(f'{original_path}/{list_splits[i]}')
        for j in range(len(list_class_dirs)):
            list_dirs = os.listdir(f'{original_path}/{list_splits[i]}/{list_class_dirs[j]}')
            for k in range(len(list_dirs)):
                image = Image.open(f'{original_path}/{list_splits[i]}/{list_class_dirs[j]}/{list_dirs[k]}')
                image = transform(image)
                if image.im.bands == 3:
                    image.save(f'{save_path}/{list_splits[i]}/{j}_{k}.jpg')

    return None


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    This method computes accuracy from logits and labels
    Parameters
    ----------
    logits : torch.Tensor
        batch of logits. Dimensions: [batch, number of classes]
    labels : torch.Tensor
        batch of labels. Dimensions: [batch]
    Returns
    -------
    float
        accuracy of predictions
    """

    # compute predictions
    predictions = logits.argmax(1).type_as(labels)

    # compute accuracy from predictions
    result = predictions.eq(labels).float().mean().cpu().detach().numpy()

    return result


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior
    
    Parameters
    ----------
    seed : int
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
