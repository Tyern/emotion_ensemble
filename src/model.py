# coding: utf-8
import torch.nn as nn
import torch
import timm

VIT_MODEL = "vit_base_patch16_224"
RESNEXT_MODEL = "resnext50_32x4d"
EFFICIENT_NET_BACKBONE = "tf_efficientnetv2_b3"


def create_model(nb_classes, activation=None, reset_parameters=False, freeze_pretrained=True):
    print("start create model")
    modelA = VisionTransformer(nb_classes)
    modelB = EfficientNetV2(nb_classes)
    modelC = ResNext(nb_classes)

    if freeze_pretrained:
        for param in modelA.parameters():
            param.requires_grad_(False)

        for param in modelB.parameters():
            param.requires_grad_(False)

        for param in modelC.parameters():
            param.requires_grad_(False)

        for param in modelA.last_fc.parameters():
            param.requires_grad = True

        for param in modelB.last_fc.parameters():
            param.requires_grad = True

        for param in modelC.last_fc.parameters():
            param.requires_grad = True

    # Create ensemble model
    model = Ensemble(modelA, modelB, modelC, activation=activation, reset_parameters=reset_parameters)
    print("done create model")
    return model


# ====================================================
# ViT Model
# ====================================================
class VisionTransformer(nn.Module):
    def __init__(self, out_dim, model_name=VIT_MODEL, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.out_features
        self.last_fc = nn.Linear(n_features, out_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.last_fc(x)
        return x


# ====================================================
# ResNext Model
# ====================================================
class ResNext(nn.Module):
    def __init__(self, out_dim, model_name=RESNEXT_MODEL, pretrained=True):
        super().__init__()
        mid_layer_dim = 100
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, mid_layer_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.last_fc = nn.Linear(mid_layer_dim, out_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.last_fc(x)
        return x


# ====================================================
# EfficientNet Model
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
# ====================================================
class EfficientNetV2(nn.Module):
    def __init__(self, out_dim, backbone=EFFICIENT_NET_BACKBONE, pretrained=True):
        super(EfficientNetV2, self).__init__()
        mid_layer_dim = 100
        self.enet = timm.create_model(backbone, pretrained=pretrained)
        self.dropout = nn.Dropout(p=0.5)
        in_ch = self.enet.classifier.in_features
        self.mid_fc = nn.Linear(in_ch, mid_layer_dim)
        self.last_fc = nn.Linear(mid_layer_dim, out_dim)
        self.enet.classifier = nn.Identity()

    def forward(self, x):
        x = self.enet(x)
        x = self.mid_fc(x)
        x = self.dropout(x)
        x = self.last_fc(x)
        return x


class Ensemble(nn.Module):
    def __init__(self, vit, resnext, efficientnet, activation=None, reset_parameters=False, nb_classes=2):
        super(Ensemble, self).__init__()
        self.modelA = vit
        self.modelB = resnext
        self.modelC = efficientnet
        self.alpha = torch.nn.Parameter(torch.rand(1))
        self.beta = torch.nn.Parameter(torch.rand(1))
        self.gamma = torch.nn.Parameter(torch.rand(1))
        self.activation = activation
        if reset_parameters:
            self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.alpha, gain=gain)
        nn.init.xavier_normal_(self.beta, gain=gain)
        nn.init.xavier_normal_(self.gamma, gain=gain)

    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x2 = self.modelB(x.clone())
        x3 = self.modelC(x)
        out = self.alpha * x1 + self.beta * x2 + self.gamma * x3
        if self.activation:
            out = self.activation(out)
        return out


class Ensemblev2(nn.Module):
    def __init__(self, vit, resnext, efficientnet, activation=torch.sigmoid, nb_classes=2):
        super(Ensemblev2, self).__init__()
        self.modelA = vit
        self.modelB = resnext
        self.modelC = efficientnet
        self.classifier = nn.Linear(6, nb_classes)
        self.activation = activation

    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        # x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x.clone())
        x3 = self.modelC(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.classifier(nn.functional.relu(out))
        return out
