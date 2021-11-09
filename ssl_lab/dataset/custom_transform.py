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


class RandCropResizeRange(Transformer):
    def __init__(self, size, scale=[0.4, 1.0]) -> None:
        super().__init__()
        assert isinstance(scale, (tuple, list))
        self.resize = tf.Resize(size)
        self.scale = scale
    
    @property
    def crop_scale(self) -> float:
        return np.random.uniform(*self.scale)

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        _, H, W = inp.shape
        scale = self.crop_scale
        crop_h = int(scale * H)
        crop_w = int(scale * W)
        inp = tf.crop.random_crop(inp, [crop_h, crop_w])
        inp = self.resize(inp)
        return inp


class RandCropResizeRangeNSQ(Transformer):
    def __init__(self, size, scale=[0.4, 1.0]) -> None:
        super().__init__()
        assert isinstance(scale, (tuple, list))
        self.resize = tf.Resize(size)
        self.scale = scale
    
    @property
    def crop_scale(self) -> float:
        return np.random.uniform(*self.scale)

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        _, H, W = inp.shape
        scale = self.crop_scale
        crop_h = int(scale * H)
        crop_w = int(scale * W)
        inp = tf.crop.random_crop(inp, min(crop_h, crop_w))
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


class TwoCropsTransform(Transformer):
    """Take two random crops of one image"""
    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]
