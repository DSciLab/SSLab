from typing import Optional
from mlutils import Trainer
from cfg import Opts
from cvutils import transform as tf
from torch.functional import Tensor


class BaseTrainer(Trainer):
    def __init__(
        self,
        opt: Opts,
        device_id: Optional[int]=None
    ) -> None:
        super().__init__(opt, device_id=device_id)
        self.to_255 = tf.DeNormalize(
            mean=[189.47680262, 185.68525998, 177.09843632],
            std=[39.60534874, 39.47619922, 37.76661493])

    def show_images(self, title: str, images: Tensor) -> None:
        images = self.to_255(images)
        self.dashboard.add_image(title, images, rgb=True)
