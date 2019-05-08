"""
Base metrics are the base classes which represent the metrics supplied with torchbearer. They all use the
:func:`.default_for_key` decorator so that they can be accessed in the call to :class:`.torchbearer.Model` via the
following strings:

- '`acc`' or '`accuracy`': The :class:`.DefaultAccuracy` metric
- '`binary_acc`' or '`binary_accuracy`': The :class:`.BinaryAccuracy` metric
- '`cat_acc`' or '`cat_accuracy`': The :class:`.CategoricalAccuracy` metric
- '`top_5_acc`' or '`top_5_accuracy`': The :class:`.TopKCategoricalAccuracy` metric
- '`top_10_acc`' or '`top_10_accuracy`': The :class:`.TopKCategoricalAccuracy` metric with k=10
- '`mse`': The :class:`.MeanSquaredError` metric
- '`loss`': The :class:`.Loss` metric
- '`epoch`': The :class:`.Epoch` metric
- '`lr`': The :class:`.LR` metric
- '`roc_auc`' or '`roc_auc_score`': The :class:`.RocAucScore` metric
"""
import torch.nn as nn
import torch.nn.functional as F

import torchbearer
from torchbearer.metrics import default_for_key, Metric, CategoricalAccuracy, MeanSquaredError, BinaryAccuracy

try:
    __loss_map__ = {
        # NN
        nn.CrossEntropyLoss.__name__: CategoricalAccuracy,
        nn.NLLLoss.__name__: CategoricalAccuracy,
        nn.MSELoss.__name__: MeanSquaredError,
        nn.BCELoss.__name__: BinaryAccuracy,
        nn.BCEWithLogitsLoss.__name__:  BinaryAccuracy,
        # Functional
        F.cross_entropy.__name__: CategoricalAccuracy,
        F.nll_loss.__name__: CategoricalAccuracy,
        F.mse_loss.__name__: MeanSquaredError,
        F.binary_cross_entropy.__name__: BinaryAccuracy,
        F.binary_cross_entropy_with_logits.__name__: BinaryAccuracy
    }
except AttributeError:  # Thrown when building the docs with mocked pytorch
    __loss_map__ = {}


@default_for_key('accuracy')
@default_for_key('acc')
class DefaultAccuracy(Metric):
    """The default accuracy metric loads in a different accuracy metric depending on the loss function or criterion in
    use at the start of training. Default for keys: `acc`, `accuracy`. The following bindings are in place for both nn
    and functional variants:

    - cross entropy loss -> :class:`.CategoricalAccuracy` [DEFAULT]
    - nll loss -> :class:`.CategoricalAccuracy`
    - mse loss -> :class:`.MeanSquaredError`
    - bce loss -> :class:`.BinaryAccuracy`
    - bce loss with logits -> :class:`.BinaryAccuracy`
    """
    def __init__(self):
        super(DefaultAccuracy, self).__init__('placeholder')  # Don't set yet, wait for reset

        self.metric = CategoricalAccuracy()  # Default to CategoricalAccuracy
        self.name = self.metric.name
        self._loaded = False
        self._train = True

    def train(self):
        self._train = True
        return self.metric.train()

    def eval(self, data_key=None):
        self._train = False
        return self.metric.eval(data_key=data_key)

    def process(self, *args):
        return self.metric.process(*args)

    def process_final(self, *args):
        return self.metric.process_final(*args)

    def reset(self, state):
        if not self._loaded:
            criterion = state[torchbearer.CRITERION]

            name = None

            if hasattr(criterion, '__name__'):
                name = criterion.__name__
            elif hasattr(criterion, '__class__'):
                name = criterion.__class__.__name__

            if name is not None and name in __loss_map__:
                self.metric = __loss_map__[name]()
                self.name = self.metric.name
                if self._train:
                    self.metric.train()
                else:
                    self.metric.eval(data_key=state[torchbearer.DATA])

            self._loaded = True

        return self.metric.reset(state)
