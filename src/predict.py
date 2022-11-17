# coding: utf-8
import warnings
import pickle
import time
import os
from model import *
import glob
import math
import sys

import pandas as pd
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup path to data folder
image_path = "../Images"
label_path = "../Labels"

test_label_path = os.path.join(label_path, "ground_truth_test.csv")

test_df = pd.read_csv(test_label_path)

image_size = (224, 224)
data_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# model create
model = create_model(len(labels)  , activation=torch.sigmoid, reset_parameters=False, freeze_pretrained=True)
weight = torch.load(sys.argv[1], device)
model.load_state_dict(weight)
model.to(device)

if len(sys.argv) > 2:
    image_path_list = sys.argv[2:]
    image_num = len(image_path_list)

    col = int(image_num**(1/2))
    row = math.ceil(image_num / col)
    fig = plt.figure(figsize=(9, 8))

    for i in range(image_num):
        ax = fig.add_subplot(row, col, i+1)

        im = Image.open(image_path_list[i])
        im_tensor = data_transforms(im).unsqueeze(0).to(device)
        with torch.inference_mode():
            label_idx = np.argmax(model(im_tensor).to("cpu")).item()

        ax.imshow(im)
        ax.axis("off")
        ax.set_title(labels[label_idx])
    plt.tight_layout()
    plt.show()
else:
    iterow = list(test_df.iterrows())
    # Copy image to test dataset dir
    for idx, row in tqdm(iterow):
        test_image_path = os.path.join(image_path, row["image"])

        im = Image.open(test_image_path)
        im_tensor = data_transforms(im).unsqueeze(0).to(device)
        with torch.inference_mode():
            label = np.argmax(model(im_tensor).to("cpu")).item()

        test_df.loc[test_df["image"]==row["image"], "label"]=label

    test_df = test_df.astype({'label': 'int32'})

test_df.to_csv("submission_result.csv", index=False)
