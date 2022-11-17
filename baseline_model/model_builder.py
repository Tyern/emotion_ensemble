import torch
from torch import nn


def create_efficient_net(
        model_size: int, 
        device: str,
        output_shape: int):
    """
    Using the torchvision fine tune model to output a trainable model
    which we can use for transfer learning.
    Weight will be use at default.

    Input:
      model_size: int: from 0 to 7
      device: str: "cpu", "mps", "cuda"

    Output:
      torchvision.models.efficientnet_b{model_size}
    """

    assert model_size in tuple(range(8)), "model size not valid, only accept 0 - 7"

    weight_base = eval(f"torchvision.models.EfficientNet_B{model_size}_Weights")
    model_base = eval(f"torchvision.models.efficientnet_b{model_size}")

    weights = weight_base.DEFAULT
    model = model_base(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False
    
    model.classifier = torch.nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=output_shape,
            bias=True
        ).to(device)
    )
    model.name = f"effnetb{model_size}"

    print(f"[INFO] Built model with name: {model.name}")
    return model
