import math
from typing import Optional
from cfg import Opts
from torch import Tensor
from torch.nn import functional as F
from ..base import BaseTrainer


class MoCoV3BaseTrainer(BaseTrainer):
    def __init__(
        self,
        opt: Opts,
        device_id: Optional[int]=None
    ) -> None:
        super().__init__(opt, device_id=device_id)

    def adjust_moco_momentum(self) -> float:
        """Adjust moco momentum based on current epoch"""
        total_steps = self.epochs * len(self.train_loader)
        m = 1. - 0.5 * (1. + math.cos(math.pi * self.step / total_steps))\
            * (1. - self.opt.moco_m)
        return m

    @property
    def moco_m(self) -> float:
        if self.opt.moco_m_cos:
            return self.adjust_moco_momentum()
        else:
            return self.opt.moco_m

    def logit_to_pred(self, logit) -> Tensor:
        return F.softmax(logit, dim=1)
        # return pred.argmax(1)


class ResNetClsBaseTrainer(BaseTrainer):
    def __init__(
        self,
        opt: Opts,
        device_id: Optional[int]=None
    ) -> None:
        super().__init__(opt, device_id=device_id)

    def logit_to_pred(self, logit):
        return F.softmax(logit, dim=1)
        # return pred.argmax(1)


class ViTClsBaseTrainer(BaseTrainer):
    def __init__(
        self,
        opt: Opts,
        device_id: Optional[int]=None
    ) -> None:
        super().__init__(opt, device_id=device_id)

    def logit_to_pred(self, logit):
        return F.softmax(logit, dim=1)
        # return pred.argmax(1)
