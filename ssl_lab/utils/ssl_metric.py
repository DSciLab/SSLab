import numpy as np
import copy
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.optim import SGD
from cfg import Opts
from mlutils import Log


class LogisticRegression(nn.Module):
    def __init__(self, in_dim, num_classes, mlp_dim=None):
        super().__init__()
        if mlp_dim is None: mlp_dim = in_dim
        self.mlp = self._build_mlp(2, in_dim, mlp_dim, num_classes)

    def _build_mlp(
        self,
        num_layers: int,
        input_dim: int,
        mlp_dim: int,
        output_dim: int,
    ):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))

        return nn.Sequential(*mlp)

    def forward(self, feature: Tensor):
        return self.mlp(feature)


class LRTrainer:
    def __init__(self, opt: Opts, in_dim: int) -> None:
        self.model = LogisticRegression(
            in_dim,
            num_classes=opt.num_classes
        )
        self.optimizer = SGD(
            self.model.parameters(),
            lr=opt.ssl_metric_lr,
            momentum=opt.ssl_metric_momentum
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.epochs = opt.ssl_metric_epochs

    def cuda(self):
        self.model = self.model.cuda()
        return self

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def fit(self, train_feat: Tensor, train_target: Tensor) -> None:
        for _ in range(self.epochs):
            logits = self.model(train_feat)
            loss = self.loss_fn(logits, train_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, eval_feat: Tensor) -> Tensor:
        return self.model(eval_feat)

    def fit_predict(
        self,
        train_feat: Tensor,
        train_target: Tensor,
        eval_feat: Tensor
    ) -> Tensor:
        self.fit(train_feat, train_target)
        return self.predict(eval_feat)


class Meter:
    def __init__(self) -> None:
        self._data = []

    def reset(self) -> None:
        self._data = []

    def append(self, item: float) -> None:
        self._data.append(item)

    def __str__(self) -> str:
        data = np.array(self._data)
        cnt_ = len(data)
        max_ = np.max(data)
        min_ = np.min(data)
        avg_ = np.mean(data)
        latest_ = data[-1]
        return (f'[SSL Metric] | latest {latest_:.5f} | max {max_:.5f} | '
                f'min {min_:.5f} | avg {avg_:.5f} | cnt {cnt_} |')

    __repr__ = __str__


class SSLMetric:
    def __init__(self, opt):
        super().__init__()
        self._lr_trainer = None
        self.num_classes = opt.num_classes
        self.available = False
        self.device = None
        self.meter = Meter()
        self.opt = opt
        self.in_dim = None
        self.train_feats = []
        self.train_targets = []
        self.eval_feats = []
        self.eval_targets = []
        self._score = None
        self._best_score = -np.Inf

    def reset(self):
        self.train_feats = []
        self.train_targets = []
        self.eval_feats = []
        self.eval_targets = []
        self._score = None

    def _get_lr_trainer(self):
        assert self.in_dim is not None
        if self._lr_trainer is None:
            self._lr_trainer = LRTrainer(self.opt, self.in_dim)
        return copy.deepcopy(self._lr_trainer).to(self.device)

    def update_train(self, feature: Tensor, target: Tensor):
        self.available = True
        self.device = feature.device
        self.in_dim = feature.size(1)
        self.train_feats.append(feature.clone().detach())
        self.train_targets.append(target.clone().detach())

    def update_eval(self, feature: Tensor, target: Tensor):
        self.eval_feats.append(feature.clone().detach())
        self.eval_targets.append(target.clone().detach())

    @staticmethod
    def one_hot(inp: Tensor, num_classes: int) -> Tensor:
        batch_size = inp.size(0)
        out = torch.zeros(
            (batch_size, num_classes),
            dtype=torch.long,
            device=inp.device
        )
        out.scatter_(1, inp.view(batch_size, 1), 1)
        return out

    def cal_acc(self, pred: Tensor, target: Tensor) -> float:
        pred = pred.type(torch.long)
        target = target.type(torch.long)

        tp = ((pred == 1) & (target == 1)).type(torch.float).sum()
        tn = ((pred == 0) & (target == 0)).type(torch.float).sum()
        return ((tp + tn) / (pred.size(0) * self.num_classes)).item()

    @property
    def score(self):
        if self._score is None:
            assert len(self.train_feats) > 0 or\
                   len(self.train_targets) > 0 or\
                   len(self.eval_feats) > 0 or\
                   len(self.eval_targets) > 0
            
            train_feats = torch.cat(self.train_feats, dim=0)
            train_targets = torch.cat(self.train_targets, dim=0)
            eval_feats = torch.cat(self.eval_feats, dim=0)
            eval_targets = torch.cat(self.eval_targets, dim=0)

            lr_trainer = self._get_lr_trainer()
            pred_logit = lr_trainer.fit_predict(
                train_feats,
                train_targets,
                eval_feats
            )
            pred = F.softmax(pred_logit, dim=1).argmax(1)
            pred = self.one_hot(pred, self.num_classes)
            eval_targets = self.one_hot(eval_targets, self.num_classes)
            self._score = self.cal_acc(pred, eval_targets)
            self.meter.append(self._score)
        return self._score

    @property
    def best(self) -> bool:
        return self.score > self._best_score

    def report(self):
        self.score
        Log.info(self.meter)
