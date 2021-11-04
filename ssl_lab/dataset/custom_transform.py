import numpy as np
from cvutils.transform.base import Transformer
from cvutils import transform as tf


class RandCropResize(Transformer):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.resize = tf.Resize(size)

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        C, H, W = inp.shape
        crop_size = min(H, W)
        inp = tf.crop.random_crop(inp, crop_size)
        inp = self.resize(inp)
        return inp


class CenterCropResize(Transformer):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.resize = tf.Resize(size)

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        C, H, W = inp.shape
        crop_size = min(H, W)
        inp = tf.crop.center_crop(inp, crop_size)
        inp = self.resize(inp)
        return inp
