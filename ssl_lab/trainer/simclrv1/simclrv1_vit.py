from cfg import Opts
from mlutils import gen, mod
from torch.functional import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch import nn
import torch
import time

from mlutils.inspector import Inspector

from .base import SimCLRV1BaseTrainer
from ssl_lab.network.vit.vit import ViT
from ssl_lab.network.simclrv1.simclrv1 import SimCLR_ViT


__all__ = ['SimCLRV1ViTSSLTrainer']


@mod.register('arch')
class SimCLRV1ViTSSLTrainer(SimCLRV1BaseTrainer):
    @gen.synchrony
    def __init__(self, opt: Opts) -> None:
        super().__init__(opt)
        net = ViT(opt)
        net = SimCLR_ViT(net)
        # print(net)
        self.optimizer = SGD(
            net.parameters(), lr=opt.lr, momentum=0.9,
            weight_decay=opt.get('weight_decay', 1.0e-4))
        self.scheduler = StepLR(self.optimizer, 10, 0.98)
        self.net = yield self.to_gpu(net)
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_model(self.net)

    @gen.detach_cpu
    @gen.synchrony
    def train_step(self, item):
        images1, images2, labels = item
        images1 = yield self.to_gpu(images1)
        images2 = yield self.to_gpu(images2)
        labels = yield self.to_gpu(labels)
        labels = labels.type(torch.int64)
        batch_size = images1.size(0)

        self.optimizer.zero_grad()
        feat_1, out_1 = self.net(images1)
        feat_2, out_2 = self.net(images2)
        self.ssl_metric.update_train(feat_1, labels)

        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        self.optimizer.step()

        # preds = self.logit_to_pred(logits)
        return loss, None, labels

    @gen.detach_cpu
    @gen.synchrony
    def eval_step(self, item):
        images1, images2, labels = item
        images1 = yield self.to_gpu(images1)
        images2 = yield self.to_gpu(images2)
        labels = yield self.to_gpu(labels)
        labels = labels.type(torch.int64)
        batch_size = images1.size(0)

        feat_1, out_1 = self.net(images1)
        feat_2, out_2 = self.net(images2)
        self.ssl_metric.update_eval(feat_1, labels)

        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        self.show_images('eval_image', images1)
        return loss, None, labels

    @gen.synchrony
    def inference(self, inp: Tensor) -> Tensor:
        inp = yield self.to_gpu(inp)

        if inp.ndim == 3:
            inp = inp.unsqueeze(0)

        with torch.no_grad():
            logits = self.net(inp)

        self.show_images('inference_image', inp)
        preds = self.logit_to_pred(logits)
        preds = yield self.to_cpu(preds.detach())
        return preds

    def on_epoch_end(self) -> None:
        self.scheduler.step()
        super().on_epoch_end()
