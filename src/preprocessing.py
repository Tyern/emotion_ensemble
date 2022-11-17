# coding: utf-8
from __future__ import print_function, division

from functools import reduce

import torch
from PIL import Image, ImageStat
from torchvision import datasets, models, transforms
import os
import operator


def total(l):
    means = [item.mean for item in l]
    stds = [item.stddev for item in l]
    return reduce(operator.__add__, means), reduce(operator.__add__, stds)


def open_image(folder: str, image_type):
    print(f"Opening image in {folder}")
    read_images = []
    for path, subdirs, files in os.walk(folder):
        for name in files:
            if name.__contains__(image_type):
                read_images.append(Image.open(os.path.join(path, name)).copy())
    return read_images


def load_data(data_dir: str, batch_size: int, num_workers: int, image_size: (int, int), image_type="png", test_mode=False):
    """
    PIL     (height x width x channel)
    pytorch (channel x height x width)
    """
    if test_mode:
        # For create test dataset
        data_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        image_datasets = datasets.ImageFolder(data_dir, data_transforms)
        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=len(image_datasets), shuffle=False, num_workers=num_workers)
        dataset_sizes = len(image_datasets)
        class_names = ["Test"]

    else:
        # Data augmentation and normalization for training
        # Just normalization for validation
        train_stats = [ImageStat.Stat(img) for img in open_image(os.path.join(data_dir, "train"), image_type)]
        val_stats = [ImageStat.Stat(img) for img in open_image(os.path.join(data_dir, "val"), image_type)]
        train_mean, train_std = total(train_stats)
        val_mean, val_std = total(val_stats)

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[train_mean[2], train_mean[0], train_mean[1]],
                                    #  std=[train_std[2], train_std[0], train_std[1]]),
            ]),
            'val': transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[val_mean[2], val_mean[0], val_mean[1]],
                                    #  std=[val_std[2], val_std[0], val_std[1]]),
            ])
        }
        # For create train/val dataset
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
            for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x])
                         for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names


if __name__ == '__main__':
    dataloaders, dataset_sizes, class_names = load_data("./data", 4, 4, (384, 384))
    print("dataloaders: ", dataloaders)
    print("dataset_sizes: ", dataset_sizes)
    print("class_names: ", class_names)
