import os
from torch import _test_serialization_subcmul
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform,
    batch_size: int,
    num_workers: int = NUM_WORKERS
) -> DataLoader:
    """
    Create DataLoader for batch learning

    :return:
    @ train_dataloader
    @ test_dataloader
    @ class_names
    """
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=transform,
        target_transform=None)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_dataloader = None
    if test_dir:
        test_dataset = datasets.ImageFolder(
            test_dir,
            transform=transform
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    return train_dataloader, test_dataloader, train_dataset.classes
