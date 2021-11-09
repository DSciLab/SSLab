# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.nn as nn
from torch.nn import functional as F


class SimCLR(nn.Module):
    def __init__(self, base_encoder, dim=128, mlp_dim=4096):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()
        self.encoder = base_encoder
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.fc(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class SimCLR_ResNet(SimCLR):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.encoder.backbone.fc.weight.shape[1]
        # remove original fc layer
        del self.encoder.backbone.fc
        self.encoder.backbone.fc = nn.Identity()

        # projectors
        self.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)


class SimCLR_ViT(SimCLR):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.encoder.mlp_head[1].weight.shape[1]
        # remove original fc layer
        del self.encoder.mlp_head
        self.encoder.mlp_head = nn.Identity()

        # projectors
        self.fc = self._build_mlp(3, hidden_dim, mlp_dim, dim)


class SimCLR_ResNet_Classifier(nn.Module):
    def __init__(self, opt, ssl_model):
        super().__init__()
        hidden_dim = ssl_model.encoder.backbone.fc[0].weight.shape[1]
        fc = nn.Linear(hidden_dim, opt.num_classes)
        self.encoder = ssl_model.encoder
        self.fc = fc

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)


class SimCLR_ViT_Classifier(nn.Module):
    def __init__(self, opt, ssl_model):
        super().__init__()
        hidden_dim = ssl_model.encoder.mlp_head[0].weight.shape[1]
        fc = nn.Linear(hidden_dim, opt.num_classes)
        self.encoder = ssl_model.encoder
        self.fc = fc

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)
