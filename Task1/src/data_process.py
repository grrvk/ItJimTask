import glob
import mimetypes
import os

import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# transforms to tensor and normalization for MNIST
img_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,)),
     ])


def load_mnist_dataset(transform=img_transforms):
    """
    Downloads (if not downloaded) and creates train and test datasets for MNIST data
    Parameters:
        transform(torchvision.transforms): transformations for datasets
    Return:
        train_dataset(torchvision.Dataset): Train dataset
        test_dataset(torchvision.Dataset): Test dataset
    """
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def get_dataloader(dataset, batch_size):
    """
    Creates dataloader for dataset
    Parameters:
        dataset(torchvision.Dataset): dataset
        batch_size(int): size of batch for dataloader
    Return:
        dataloader(DataLoader): Dataset dataloader
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_image_paths(folder_path):
    """
    Get paths to all images in the folder
    Parameters:
        folder_path(str): path to the folder to look for images in
    Return:
        image_paths(list): list of full paths to the images
    """
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    return image_paths


def get_input_paths(data_path):
    """
    Determine type of custom input object by given path, look for images
    Parameters:
        data_path(str): users path to data for prediction
    Return:
        image_paths(list): list of paths to images, if there are any
    """
    assert os.path.exists(data_path), 'Path is invalid'
    mime_type, _ = mimetypes.guess_type(data_path)

    if os.path.isdir(data_path):
        return get_image_paths(data_path)
    elif mime_type and mime_type.startswith("image/"):
        return [data_path]

    print('File is not an image nor a folder')
    return []


def prepare_custom_input(data_path, target_image_size=(28, 28)):
    """
    Prepare data from user input path for prediction
    Parameters:
        data_path(str): users path to data for prediction
    Return:
        image_paths(list): list of paths to images, if there are any
        images(torch.Tensor): list of images as pytorch Tensors
    """
    image_paths = get_input_paths(data_path)
    images = []

    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Unable to read {image_path}")
            continue
        img = cv2.resize(img, target_image_size, interpolation=cv2.INTER_AREA)
        images.append(img)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,)),
        ])
    images = [transform(img) for img in images]
    images = torch.stack(images)

    return image_paths, images


def print_predictions(image_paths, predictions):
    """
    Prepare data from user input path for prediction
    Parameters:
        image_paths(str): paths to images for which predictions were made
        predictions():
    """
    for path, label in zip(image_paths, predictions):
        print(f"Image: {os.path.basename(path)}, Predicted Label: {label}")
