
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import copy
import torch.nn as nn
from torch.nn.modules import loss
from ssl_lab.utils.model import freeze


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=256, m=0.999, T=0.07, mlp_dim=4096):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.ce = nn.CrossEntropyLoss()
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)

        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp),
        #         nn.ReLU(),
        #         self.encoder_q.fc
        #     )
        #     self.encoder_k.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp),
        #         nn.ReLU(),
        #         self.encoder_k.fc
        #     )

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    # @torch.no_grad()
    # def _batch_shuffle_ddp(self, x):
    #     """
    #     Batch shuffle, for making use of BatchNorm.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]

    #     num_gpus = batch_size_all // batch_size_this

    #     # random shuffle index
    #     idx_shuffle = torch.randperm(batch_size_all).cuda()

    #     # broadcast to all gpus
    #     torch.distributed.broadcast(idx_shuffle, src=0)

    #     # index for restoring
    #     idx_unshuffle = torch.argsort(idx_shuffle)

    #     # shuffled index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    #     return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_shuffle(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support Non DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size = x.shape[0]
        # x_gather = concat_all_gather(x)
        # batch_size_all = x_gather.shape[0]

        # num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size).cuda()

        # broadcast to all gpus
        # torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        # gpu_idx = torch.distributed.get_rank()

        return x[idx_shuffle], idx_unshuffle

    # @torch.no_grad()
    # def _batch_unshuffle_ddp(self, x, idx_unshuffle):
    #     """
    #     Undo batch shuffle.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]

    #     num_gpus = batch_size_all // batch_size_this

    #     # restored index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    #     return x_gather[idx_this]

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support Non DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        # batch_size = x.shape[0]
        # x_gather = concat_all_gather(x)
        # batch_size_all = x_gather.shape[0]

        # num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        # gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x[idx_unshuffle]

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

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            im_k, idx_unshuffle = self._batch_shuffle(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k = self._batch_unshuffle(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        loss = self.ce(logits, labels)
        return loss, q


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.encoder_q.backbone.fc.weight.shape[1]
        # remove original fc layer
        del self.encoder_q.backbone.fc, self.encoder_k.backbone.fc

        # projectors
        self.encoder_q.backbone.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.encoder_k.backbone.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.encoder_q.mlp_head[1].weight.shape[1]
        # remove original fc layer
        del self.encoder_q.mlp_head, self.encoder_k.mlp_head

        # projectors
        self.encoder_q.mlp_head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.encoder_k.mlp_head = self._build_mlp(3, hidden_dim, mlp_dim, dim)


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


# utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output