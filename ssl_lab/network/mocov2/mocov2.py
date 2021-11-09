import torch
import copy
import torch.nn as nn
from torch.nn import functional as F


class MoCo(nn.Module):
    """
    an implementation of MoCo v1 + v2
    MoCo v1: https://arxiv.org/abs/1911.05722
    MoCo v1: https://arxiv.org/abs/2003.04297
    """
    def __init__(
        self,
        base_encoder: nn.Module,
        dim: int=256,
        queue_size: int=256,
        momentum: float=0.999,
        temperature: float=0.07,
        bias: bool=True,
        moco: bool=False,
        mlp_dim: int=4096,
        *args,
    ):
        super().__init__()
        self.dim = dim  # C
        self.queue_size = queue_size  # K
        self.momentum = momentum  # m
        self.temperature = temperature  # t
        self.bias = bias
        self.moco = moco
        self.args = args

        # create the queue
        self.register_buffer(
            "queue", 
            F.normalize(
                torch.randn(
                    self.queue_size,
                    self.dim,
                    requires_grad=False
                ),
                dim=1
            ) / 10
        )
        self.ptr = 0

        self.ce = nn.CrossEntropyLoss()
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        # freeze k_encoder params (for manual momentum update)
        for p_k in self.k_encoder.parameters():
            p_k.requires_grad = False

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
                # follow SimCLR's design:
                # https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    @torch.no_grad()
    def update_k_encoder_weights(self):
        """ manually update key encoder weights with momentum and no_grad"""
        # update k_encoder.parameters
        for p_q, p_k in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
            p_k.data = p_k.data*self.momentum + (1.0 - self.momentum)*p_q.data
            p_k.requires_grad = False

        # update k_fc.parameters
        for p_q, p_k in zip(self.q_fc.parameters(), self.k_fc.parameters()):
            p_k.data = p_k.data*self.momentum + (1.0 - self.momentum)*p_q.data
            p_k.requires_grad = False

    @torch.no_grad()
    def update_queue(self, k):
        """ swap oldest batch with the current key batch and update ptr"""
        batch_size = k.size(0)
        self.queue[self.ptr: self.ptr + batch_size, :] = k.detach().cpu()
        self.ptr = (self.ptr + batch_size) % self.queue_size
        self.queue.requires_grad = False

    def forward(self, q, k):
        """ moco phase forward pass """
        q_enc = self.q_encoder(q)  # queries: NxC
        q = self.q_fc(q_enc)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            k = self.k_encoder(k)  # keys: NxC
            k = self.k_fc(k)
            k = F.normalize(k, dim=1)

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,kc->nk', [q, self.queue.clone().detach()])

        self.update_k_encoder_weights()
        self.update_queue(k)


        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        # contrastive loss labels, positive logits used as ground truth
        zeros = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        loss = self.ce(logits, zeros)

        return loss, q_enc.detach()


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.encoder_q.backbone.fc.weight.shape[1]
        # remove original fc layer
        del self.encoder_q.backbone.fc, self.encoder_k.backbone.fc

        # projectors
        self.encoder_q.backbone.fc = nn.Identity()
        self.q_fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.encoder_k.backbone.fc = nn.Identity()
        self.q_fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.encoder_q.mlp_head[1].weight.shape[1]
        # remove original fc layer
        del self.encoder_q.mlp_head, self.encoder_k.mlp_head

        # projectors
        self.encoder_q.mlp_head = nn.Identity()
        self.q_fc = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.encoder_k.mlp_head = nn.Identity()
        self.q_fc = self._build_mlp(3, hidden_dim, mlp_dim, dim)


class MoCo_ResNet_Classifier(nn.Module):
    def __init__(self, opt, ssl_model):
        super().__init__()
        hidden_dim = ssl_model.encoder_q.backbone.fc[0].weight.shape[1]
        fc = nn.Linear(hidden_dim, opt.num_classes)
        self.model = ssl_model.encoder_q
        self.model.backbone.fc = fc

    def forward(self, x):
        return self.model(x)


class MoCo_ViT_Classifier(nn.Module):
    def __init__(self, opt, ssl_model):
        super().__init__()
        hidden_dim = ssl_model.encoder_q.mlp_head[0].weight.shape[1]
        fc = nn.Linear(hidden_dim, opt.num_classes)
        self.model = ssl_model.encoder_q
        self.model.mlp_head = fc

    def forward(self, x):
        return self.model(x)
