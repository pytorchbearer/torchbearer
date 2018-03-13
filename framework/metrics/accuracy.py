from framework.metrics.metrics import Lambda

import torch


def _categorical(y_true, y_pred):
    _, y_pred = torch.max(y_pred, 1)
    return y_pred == y_true


categorical = accuracy = Lambda('acc', _categorical)
