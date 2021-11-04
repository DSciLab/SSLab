from typing import List, Any, Tuple
from torch import Tensor
import numpy as np
from torch.utils.data.dataset import Dataset
from cvutils import transform as tf
from mlutils import mod, split_by_kfold, split_by_proportion, Log
from cfg import Opts

from trans_lab.utils.preprocessing import load_npy, load_pickle


class InWBaseDataset(Dataset):
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
        super().__init__()
        self.training = training
        self.original_data_source = data_source
        self.set_transformer(opt)

        if self.training:
            Log.info(f'Data size for training is {len(self)}')
        else:
            Log.info(f'Data size for validation is {len(self)}')

    def set_transformer(self, opt: Opts):
        if self.training:
            self.transform = tf.Compose([
                tf.TransposeTorch(),
                tf.Normalize(),
                # tf.RandomTransform(opt.rand_k),
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
