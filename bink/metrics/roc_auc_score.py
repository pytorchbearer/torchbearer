from bink import metrics

import sklearn.metrics
import numpy as np


class RocAucScore(metrics.EpochLambda):
    def __init__(self, one_hot_labels=True, one_hot_offset=0, one_hot_classes=10):

        def to_categorical(y):
            return np.eye(one_hot_classes, dtype='uint8')[y - one_hot_offset]

        if one_hot_labels:
            process = to_categorical
        else:
            process = lambda y: y

        super().__init__('roc_auc_score', lambda y_true, y_pred: sklearn.metrics.roc_auc_score(process(y_true.cpu().numpy()), y_pred.cpu().numpy()))
