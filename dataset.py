import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Global data directory (relative to project root)
DATA_DIR = "data"


def get_transforms(augment: bool = True):
    """Returns train and validation transform pipelines for MNIST.

    Args:
        augment: If True, applies random affine distortions (rotation, translation,
                 scale, shear) to mimic real-world handwriting variability.
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return train_transform, val_transform


def get_train_loader(batch_size: int = 64, shuffle: bool = True, augment: bool = True, **kwargs):
    """Constructs and returns a DataLoader for the training MNIST partition."""
    train_transform, _ = get_transforms(augment=augment)
    dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def get_val_loader(batch_size: int = 64, shuffle: bool = False, **kwargs):
    """Constructs and returns a DataLoader for the validation MNIST partition."""
    _, val_transform = get_transforms()
    dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=val_transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)