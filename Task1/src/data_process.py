import glob
import mimetypes
import os

import cv2
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

img_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,)),
     ])


def load_mnist_dataset(transform):
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def get_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_data():
    train_dataset, test_dataset = load_mnist_dataset(img_transforms)
    train_dataloader = get_dataloader(train_dataset, 64)
    test_dataloader = get_dataloader(test_dataset, 64)
    return train_dataloader, test_dataloader


def get_image_paths(folder_path):
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    return image_paths


def get_input_paths(data_path):
    assert os.path.exists(data_path), 'Path is invalid'
    mime_type, _ = mimetypes.guess_type(data_path)

    if os.path.isdir(data_path):
        return get_image_paths(data_path)
    elif mime_type and mime_type.startswith("image/"):
        return [data_path]

    print('File is not an image nor a folder')
    return []


def prepare_custom_input(data_path, target_image_size=(28, 28)):
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


def print_predictions(data_paths, predictions):
    for path, label in zip(data_paths, predictions):
        print(f"Image: {os.path.basename(path)}, Predicted Label: {label}")
