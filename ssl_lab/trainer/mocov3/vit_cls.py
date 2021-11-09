from cfg import Opts
from mlutils import gen, mod
from torch.functional import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch import nn
import torch

from mlutils.inspector import Inspector

from .base import ViTClsBaseTrainer
from ssl_lab.network.mocov3.mocov3 import MoCo_ViT_Classifier


__all__ = ['MoCoV3ViTClsTrainer']


@mod.register('arch')
class MoCoV3ViTClsTrainer(ViTClsBaseTrainer):
    @gen.synchrony
    def __init__(self, opt: Opts) -> None:
        super().__init__(opt)
        ssl_model = self.load_ssl_model()
        net = MoCo_ViT_Classifier(opt, ssl_model)
        # print(net)
        self.optimizer = SGD(
            net.parameters(), lr=opt.lr, momentum=0.9,
            weight_decay=opt.get('weight_decay', 1.0e-4))
        self.scheduler = StepLR(self.optimizer, 10, 0.98)
        self.net = yield self.to_gpu(net)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.eval_no_grad = False
        # self.inspector = Inspector(opt, self.net)
        # self.inspector.regist_layers('backbone.layer4.2.relu')

    @gen.detach_cpu
    @gen.synchrony
    def train_step(self, item):
        images, labels = item
        images = yield self.to_gpu(images)
        labels = yield self.to_gpu(labels)
        labels = labels.type(torch.int64)

        self.optimizer.zero_grad()
        logits = self.net(images)
        loss = self.loss_fn(logits, labels)
        loss.backward()
        self.optimizer.step()

        preds = self.logit_to_pred(logits)
        return loss, preds, labels

    @gen.detach_cpu
    @gen.synchrony
    def eval_step(self, item):
        images, labels = item
        images = yield self.to_gpu(images)
        labels = yield self.to_gpu(labels)
        labels = labels.type(torch.int64)

        logits = self.net(images)
        loss = self.loss_fn(logits, labels)

        self.show_images('eval_image', images)
        # self.inspector.inspect(images)
        # cam_image = self.inspector.show_cam_on_images(strength=1.5)[0]
        # self.show_images('cam_image', cam_image)
        # pred = torch.sigmoid(logits).clone().detach()
        # pred[pred>0.5] = 1.0
        # pred[pred<=0.5] = 0.0
        # result = (f'score: {torch.sigmoid(logits)} <br/>'
        #           f'pred: {pred} <br/>'
        #           f'label: {labels} <br/>'
        #           f'result: {pred == labels}')
        # self.dashboard.add_text('reselt', result)
        preds = self.logit_to_pred(logits)
        return loss, preds, labels

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
