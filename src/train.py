# coding: utf-8
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm

import time
import copy
import warnings

from preprocessing import load_data
from model import *
from utils import plot_training_process, save_confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]


def fine_tuning():
    data_loaders, image_datasets, class_names = load_data("../data", 32, 16, (224, 224), image_type="jpg")
    

    model_ft = create_model(len(class_names), activation=nn.Softmax(dim=1), reset_parameters=False, freeze_pretrained=True)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.01)

    # Decay LR by a factor of 0.1 every 10 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.95)
    model_ft = train_model(data_loaders, image_datasets, class_names, model_ft, criterion, optimizer_ft,
                           num_epochs=3)
    return model_ft


def train_model(data_loaders, image_datasets, class_names, model, criterion, optimizer, scheduler=None, num_epochs=25):
    dataset_sizes = {x: len(image_datasets[x])
                         for x in ['train', 'val']}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss, train_acc, val_loss, val_acc = [], [], [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                if scheduler:
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save model
    torch.save(model.state_dict(), f'{num_epochs:03}ep_acc{best_acc:.2f}.pth')

    # plot training process
    plot_training_process(train_acc, val_acc, train_loss, val_loss)

    y_preds = []
    with torch.inference_mode():
        for inputs, labels in tqdm(data_loaders["val"], desc="Making predictions"):
            # Send data and targets to target device
            inputs, labels = inputs.to(device), labels.to(device)
            # Do the forward pass
            y_logit = model(inputs)
            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())
        # Concatenate list of predictions into a tensor
        y_pred_tensor = torch.cat(y_preds)

    save_confusion_matrix(
        class_names=all_labels, 
        y_pred_tensor=y_pred_tensor,
        test_dataset=image_datasets['val']
    )

    return model


if __name__ == '__main__':
    fine_tuning()
