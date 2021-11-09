import torch
from torch import nn


def freeze(model: nn.Module) -> nn.Module:
    for _, param in model.named_parameters():
        param.requires_grad = False
    return model
