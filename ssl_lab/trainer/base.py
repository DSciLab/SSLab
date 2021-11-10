import copy
from torch import nn
from typing import Optional
from mlutils import Trainer
from cfg import Opts
from cvutils import transform as tf
from torch.functional import Tensor
from ssl_lab.utils.model_loader import ModelLoader
from ssl_lab.utils.ssl_metric import SSLMetric, SSLEuclideanMetric


class BaseTrainer(Trainer):
    def __init__(
        self,
        opt: Opts,
        device_id: Optional[int]=None
    ) -> None:
        super().__init__(opt, device_id=device_id)
        self.model_loader = ModelLoader(opt, self.saver)
        self.ssl_metric = SSLMetric(opt)
        self.ssl_eu_metric = SSLEuclideanMetric(opt)
        self.to_255 = tf.DeNormalize(
            mean=[189.47680262, 185.68525998, 177.09843632],
            std=[39.60534874, 39.47619922, 37.76661493])

    def show_images(self, title: str, images: Tensor) -> None:
        images = self.to_255(images)
        self.dashboard.add_image(title, images, rgb=True)

    def load_ssl_model(self) -> nn.Module:
        model = self.model_loader.load_model()
        return model

    def save_model(self, model):
        model = copy.deepcopy(model)
        model = model.cpu()
        self.saver.save_object(model, 'raw_model.pkl')

    @property
    def best(self) -> bool:
        if self.ssl_metric.available:
            return self.ssl_metric.best
        else:
            return super().best

    def on_epoch_end(self):
        if self.ssl_metric.available:
            self.ssl_metric.report()
            self.ssl_metric.reset()
        if self.ssl_eu_metric.available:
            self.ssl_eu_metric.report()
            self.ssl_eu_metric.reset()
        super().on_epoch_end()
