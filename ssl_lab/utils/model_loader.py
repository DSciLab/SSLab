import os
import torch
from cfg import Opts
from mlutils import Saver, Log
from mlutils.saver import load_pickle

class ModelLoader:
    def __init__(
        self,
        opt: Opts,
        saver: Saver
    ) -> None:
        self.ssl_id = opt.get('ssl_id', None)
        if self.ssl_id is not None:
            self.ssl_model = opt.ssl_model
            self.root = os.path.join(*saver.saver_dir.split('/')[:-1], self.ssl_id)

    def load_model(self):
        assert self.ssl_id is not None
        raw_model_path = os.path.join(self.root, 'raw_model.pkl')
        model = load_pickle(raw_model_path)

        weights = self.load_weight()
        model.load_state_dict(weights)
        return model

    def load_weight(self):
        assert self.ssl_id is not None
        assert self.ssl_model in ['best', 'latest'], f'{self.ssl_model} error.'
        weight_path = os.path.join(self.root, f'{self.ssl_model}_0.pth')
        state_dict = torch.load(weight_path)
        epoch = state_dict['epoch']
        Log.info(f'Load model from {epoch}_th epoch.')
        return state_dict['net']
