from typing import Mapping, Tuple
from torch.utils.data.dataloader import DataLoader
from cfg import Opts
import tqdm
import numpy as np
from mlutils import mod, init, Log
from trans_lab.trainer import *


def test(opt: Opts) -> None:
    init(opt)
    trainer = mod.get('arch', opt.arch)(opt)
    trainer.load_state()
    trainer.eval_state()

    saver = trainer.saver
    val_data_source = saver.load_object('val_data_source_0.pkl')
    dataset = mod.get('dataset', opt.dataset)(opt, val_data_source, training=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True
    )

    preds = []
    labels = []
    for image, label in tqdm.tqdm(dataloader):
        pred = trainer.inference(image)
        pred = pred.view(-1).item()
        label = label.view(-1).item()

        preds.append(pred)
        labels.append(label)

    result = {
        'preds': np.array(preds),
        'labels': np.array(labels)
    }
    result_statistic(result)
    saver.save_object(result, 'predict_result.pkl')


def result_statistic(
    result: Mapping[str, np.ndarray]
) -> None:
    tp, fp, tn, fn = confusion_matrix(result)
    acc = (tp + tn) / (tp + fp + tn + fn)
    Log.info(f'ACC: {acc}')
    Log.info(f'TP: {tp}')
    Log.info(f'FN: {fn}')
    Log.info(f'TN: {tn}')
    Log.info(f'FP: {fp}')


def confusion_matrix(
    result: Mapping[str, np.ndarray]
) -> Tuple[float, float, float, float]:
    preds = result['preds']
    labels = result['labels']
    preds = threshold_array(preds)
    labels = threshold_array(labels)

    tp = ((preds == 1.0).astype(np.float) * (labels == 1.0).astype(np.float)).sum()
    fp = ((preds == 1.0).astype(np.float) * (labels == 0.0).astype(np.float)).sum()
    tn = ((preds == 0.0).astype(np.float) * (labels == 0.0).astype(np.float)).sum()
    fn = ((preds == 0.0).astype(np.float) * (labels == 1.0).astype(np.float)).sum()

    return tp, fp, tn, fn


def threshold_array(
    inp: np.ndarray
) -> np.ndarray:
    inp = np.copy(inp)
    inp[inp>0.5] = 1.0
    inp[inp<=0.5] = 0.0
    return inp
