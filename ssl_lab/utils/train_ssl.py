from torch.utils.data.dataloader import DataLoader
from cfg import Opts
from mlutils import mod, init
from mlutils import metrics as mt
from ssl_lab.trainer import *
from ssl_lab.dataset import *


def train(opt: Opts) -> None:
    init(opt)
    trainer = mod.get('arch', opt.arch)(opt)
    # trainer.set_metrics(mt.Accuracy)

    for k, (training_set, val_set) in enumerate(get_dataset(opt)):
        training_loader = DataLoader(
            training_set,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers)
        val_loader = DataLoader(
            val_set,
            batch_size=opt.get('val_batch_size', opt.batch_size),
            shuffle=False,
            num_workers=opt.num_workers)

        # save data_source
        saver = trainer.saver
        training_data_source = training_set.original_data_source
        val_data_source = val_set.original_data_source
        saver.save_object(training_data_source, f'training_data_source_{k}.pkl')
        saver.save_object(val_data_source, f'val_data_source_{k}.pkl')

        trainer.train(training_loader, val_loader)
