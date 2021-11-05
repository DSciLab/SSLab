from typing import List, Any, Mapping, Tuple
from torch import Tensor
import numpy as np
from cvutils import transform as tf
from cvutils import imread
from mlutils import mod, Log
from cfg import Opts

from trans_lab.utils.data_split import split_by_k_fold, split_by_proportion
from trans_lab.utils.preprocessing import load_npy, load_pickle
from .base import InWBaseDataset


__all__ = ['InWClsDataset', 'InWCls224Dataset', 'InWCls448Dataset',
           'InWCls448SQDataset', 'InWCls448WDataset',
           'InWCls224SQDataset', 'InWCls256SQDataset']


class InWClsDatasetBase(InWBaseDataset):
    LABEL_MAPPING = {
        'real': 0.,
        'fake': 1.
    }

    def __init__(
        self,
        opt: Opts,
        data_source: List[Any],
        training: bool=True
    ) -> None:
        self.data_len = opt.get('data_len', -1)
        self.data_source = data_source
        super().__init__(opt, data_source, training)

    def set_transformer(self, opt: Opts):
        if self.training:
            self.transform = tf.Compose([
                tf.TransposeTorch(),
                tf.Normalize(),
                tf.RandomTransform(opt.rand_k, check=False),
                tf.Resize(opt.input_size),
                tf.ToTensor(),
            ])
        else:
            self.transform = tf.Compose([
                tf.TransposeTorch(),
                tf.Normalize(),
                tf.Resize(opt.input_size),
                tf.ToTensor(),
            ])

    @classmethod
    def from_path_to_label(cls, path: str) -> int:
        file_name = path.split('/')[-1]
        if 'real' in file_name:
            return cls.LABEL_MAPPING['real']
        elif 'fake' in file_name:
            return cls.LABEL_MAPPING['fake']
        else:
            raise ValueError(f'Unrecognized file {path}')

    @staticmethod
    def get_data_source(opt: Opts) -> List[Any]:
        meta_data = load_pickle(opt.meta_path)
        data_len = len(meta_data['real_images']) + len(meta_data['fake_images'])
        Log.info(f'Original dataset size is {data_len}.')
        k_fold = opt.get('k_fold', -1)
        if k_fold <= 0:
            Log.info(f'Training by proportion.')
            return split_by_proportion(
                meta_data,
                opt.training_proportion
            )
        else:
            Log.info(f'Training by {k_fold} fold.')
            return split_by_k_fold(
                meta_data,
                k_fold
            )

    def __len__(self) -> int:
        if self.data_len > 0 and self.training:
            return self.data_len
        else:
            return len(self.data_source)

    def __getitem__(
        self,
        index: int
    ) -> Tuple[Tensor, int, Mapping[str, int]]:
        index = index % len(self.data_source)
        image_path = self.data_source[index]
        label = self.from_path_to_label(image_path)

        image = load_npy(image_path)
        image = self.transform(image)
        return image, label


@mod.register('dataset')
class InWClsDataset(InWClsDatasetBase):
    pass


@mod.register('dataset')
class InWCls224Dataset(InWClsDatasetBase):
    def set_transformer(self, opt: Opts):
        if opt.get('on_ssl', False) and opt.get('ssl_id', None) is None:
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
        else:
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    tf.RandomTransform(opt.rand_k, check=False),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])


@mod.register('dataset')
class InWCls256Dataset(InWClsDatasetBase):
    def set_transformer(self, opt: Opts):
        if opt.get('on_ssl', False) and opt.get('ssl_id', None) is None:
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
        else:
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    tf.RandomTransform(opt.rand_k, check=False),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])


@mod.register('dataset')
class InWCls448Dataset(InWClsDatasetBase):
    def set_transformer(self, opt: Opts):
        if opt.get('on_ssl', False) and opt.get('ssl_id', None) is None:
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
        else:
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    # tf.RandomRotate(),
                    tf.ResizeRandomCenterCroPad(
                        opt.input_size,
                        rand_range=[0.5, 1.5]
                    ),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    tf.RandomTransform(opt.rand_k, check=False),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])


@mod.register('dataset')
class InWCls448SQDataset(InWClsDatasetBase):
    def set_transformer(self, opt: Opts):
        if opt.get('on_ssl', False) and opt.get('ssl_id', None) is None:
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
        else:
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    tf.RandomTransform(opt.rand_k, check=False),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])


@mod.register('dataset')
class InWCls224SQDataset(InWClsDatasetBase):
    def set_transformer(self, opt: Opts):
        if opt.get('on_ssl', False) and opt.get('ssl_id', None) is None:
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
        else:
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    tf.RandomTransform(opt.rand_k, check=False),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])


@mod.register('dataset')
class InWCls256SQDataset(InWClsDatasetBase):
    def set_transformer(self, opt: Opts):
        if opt.get('on_ssl', False) and opt.get('ssl_id', None) is None:
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
        else:
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    tf.RandomTransform(opt.rand_k, check=False),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])


@mod.register('dataset')
class InWCls448WDataset(InWClsDatasetBase):
    def set_transformer(self, opt: Opts):
        if opt.get('on_ssl', False) and opt.get('ssl_id', None) is None:
            # ssl
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])
        else:
            # cls
            if self.training:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    tf.RandomTransform(opt.rand_k, check=False),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])
            else:
                self.transform = tf.Compose([
                    tf.TransposeTorch(),
                    tf.Normalize(mean=opt.mean, std=opt.std),
                    # tf.Resize(opt.input_size),
                    tf.ToTensor(),
                ])
