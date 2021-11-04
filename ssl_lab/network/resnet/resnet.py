from torch import nn
from torch import Tensor
import torch
from torchvision.models.resnet import *

from cfg import Opts


_resnet_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2
}


class ResNet(nn.Module):
    expansion_dict = {
        'resnet18': 1,
        'resnet34': 1,
        'resnet50': 4,
        'resnet101': 4,
        'resnet152': 4,
        'resnext50_32x4d': 4,
        'resnext101_32x8d': 4,
        'wide_resnet50_2': 4,
        'wide_resnet101_2': 4
    }

    def __init__(self, opt: Opts, backbone: str) -> None:
        super().__init__()
        self.backbone = _resnet_dict[backbone](**opt.resnet_kws)
        self.backbone.fc = nn.Linear(512 * self.get_expansion(backbone),
                                     opt.num_classes)

    @classmethod
    def get_expansion(cls, backbone: str) -> int:
        return cls.expansion_dict[backbone]

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        # x = torch.flatten(x, 0)
        return x
