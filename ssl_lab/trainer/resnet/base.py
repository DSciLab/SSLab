from typing import Optional
from cfg import Opts
from torch.nn import functional as F
from ..base import BaseTrainer


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
