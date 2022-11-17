import time
import torch
import torchvision
assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"

import os
import sys
import glob
import random

import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms

import data_setup, engine, utils, model_builder

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

train_dir = f"../data/train"
test_dir = f"../data/test"
val_dir = f"../data/val"

EPOCHS = 100
BATCH_SIZE = 16

if __name__ == "__main__":
    normalize_transforms = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # manual transform should be considered matching with
    # transfer learning model preprocessing step
    manual_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.TrivialAugmentWide(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transforms
    ])

    train_dataloader, val_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=val_dir,
        transform=manual_transform,
        batch_size=BATCH_SIZE,
        num_workers=1
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    ## Use efficient net
    model = model_builder.create_efficient_net(
        model_size=1,
        device=device,
        output_shape=len(class_names),
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Start training
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    writer = utils.create_writer(
        experiment_name="ExperimentTracking-startup",
        model_name=model.name,
        extra=f"{EPOCHS}-epoch"
    )

    start = time.perf_counter()

    results = engine.train(
        model = model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=device,
        writer=writer,
        n_epoch_per_save=100
    )

    end = time.perf_counter()
    print(f"[INFO] Execution time: {end - start : .3f}s, device: {device}")
    print()

    writer.close()
