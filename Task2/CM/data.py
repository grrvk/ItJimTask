import torch
from PIL import Image
from datasets import load_dataset
from torchvision.transforms import v2


# transforms for the images
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
])


def prepare_data(datapath):
    """
    Create valid dataset dict
    Parameters:
        datapath(str): path to dataset folder
    Returns:
        dataset(datasets.DatasetDist): valid dataset dict
    """
    def transforms(examples):
        examples['pixel_values'] = [transform(image.convert("RGB")) for image in examples['image']]
        return examples

    dataset = load_dataset(datapath)
    dataset['train'].set_transform(transforms)
    dataset['validation'].set_transform(transforms)
    dataset['test'].set_transform(transforms)
    return dataset

def prepare_image(image_path):
    """
    Load user inputted image from path and prepare for classification
    """
    image = Image.open(image_path)
    image = transform(image)
    return image


if __name__ == '__main__':
    dataset = prepare_data('/Users/vika/Desktop/dataset')
    print(dataset)

