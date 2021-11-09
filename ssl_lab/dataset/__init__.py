from typing import Generator
from cfg import Opts
from mlutils import mod
from .inw_cls import *
from .inw_ssl import *
from .inw_cls_crop import *



def get_dataset(opt: Opts) -> Generator:
    dataset_cls = mod.get('dataset', opt.dataset)
    training_data, val_data = dataset_cls.get_data_source(opt)

    training_set = dataset_cls(opt, training_data, training=True)
    val_set = dataset_cls(opt, val_data, training=False)

    yield training_set, val_set
