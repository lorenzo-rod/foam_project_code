import torchvision.models as models
import timm
import torch.nn as nn
from typing import OrderedDict
import sys


def design_model(model_to_use, hidden_size, layers_to_freeze, num_fc, n_outputs, dropout_prob):
    """Generates neural network from some pretrained architectures

    Args:
        model_to_use (str): desired pretrained model to use
        hidden_size (int): number of neurons in the linear layers
        device (str): device in which to carry operations (cuda or cpu)
        layers_to_freeze (int): number of layers to freeze from the model
        num_fc (int): number of linear layers after the convolutional layers
        n_outputs (int): number of outputs the nn will have
        dropout_prob (float): dropout probability

    Raises:
        ValueError: model to use is not supported
        ValueError: num_fc is not valid (at least 2 linear layers at the end)

    Returns:
        nn.Module: neural network that has the specifications provided in the input
    """
    # Select pretrained model
    if model_to_use == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_to_use == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_to_use == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_to_use == "wide_resnet_50_2":
        model = models.wide_resnet50_2(pretrained=True)
    elif model_to_use == "xception":
        model = timm.create_model('xception', pretrained=True)
    elif model_to_use == "googlenet":
        model = models.googlenet(pretrained=True)
    else:
        raise ValueError(model_to_use + " is not a supported model")


    if num_fc < 2:
        raise ValueError(f"{num_fc} must be greater than 1")


    if model_to_use != "custom":
        num_ftrs = model.fc.in_features # number of features in the input of the final layers
        limit = 0
        # for loop freezes first layers_to_freeze layers
        for child in model.children():
            limit += 1
            if limit < layers_to_freeze:
                for param in child.parameters():
                    param.requires_grad = False
        fc = [] # array that contains the final layers in the model
        fc.append(("Dropout 0", nn.Dropout(p=dropout_prob)))
        fc.append(("linear 0", nn.Linear(num_ftrs, hidden_size)))
        fc.append(("ReLU 0", nn.ReLU()))
        fc.append(("Dropout 1", nn.Dropout(p=dropout_prob)))
        # for loop that appends all the linear layers and activation functions in fc
        # Tuples indicating the type of layer and its number are added because it is cast into an
        # ordered dict afterwards
        for i in range(num_fc - 2):
            fc.append(("linear " + str(i + 1), nn.Linear(hidden_size, hidden_size)))
            fc.append(("ReLU " + str(i + 1), nn.ReLU()))
            fc.append(("Dropout " + str(i + 2), nn.Dropout(p=dropout_prob)))
        fc.append(("linear " + str(num_fc - 1), nn.Linear(hidden_size, n_outputs)))
        # list is cast into an OrderedDict
        fc = OrderedDict(fc)
        # Layers are set into the model
        model.fc = nn.Sequential(fc)

    return model

