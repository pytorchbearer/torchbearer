import torch

from bink.callbacks.callbacks import Callback
import os


class _Checkpointer(Callback):
    def __init__(self, fileformat, pickle_module=torch.serialization.pickle, pickle_protocol=torch.serialization.DEFAULT_PROTOCOL):
        super().__init__()
        self.fileformat = fileformat

        self.pickle_module = pickle_module
        self.pickle_protocol = pickle_protocol

        self.most_recent = None

        if fileformat.__contains__(os.sep) and not os.path.exists(os.path.dirname(fileformat)):
            os.makedirs(os.path.dirname(fileformat))

    def save_checkpoint(self, model_state, overwrite_most_recent=False):
        state = {}
        state.update(model_state)
        state.update(model_state['metrics'])

        filepath = self.fileformat.format(**state)

        torch.save(model_state['self'].state_dict(), filepath, pickle_module=self.pickle_module, pickle_protocol=self.pickle_protocol)

        if self.most_recent is not None and overwrite_most_recent:
            os.remove(self.most_recent)

        self.most_recent = filepath


def ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.pt',
        monitor='val_loss', save_best_only=False, mode='auto', period=1, min_delta=0):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled any values from state.
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename. The torch model will be saved to filename.pt
    and the binkmodel state will be saved to filename.bink.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """
    if save_best_only:
        check = Best(filepath, monitor, mode, period, min_delta)
    else:
        check = Interval(filepath, period)

    return check


class MostRecent(_Checkpointer):
    def __init__(self, filepath='model.{epoch:02d}-{val_loss:.2f}.pt', pickle_module=torch.serialization.pickle, pickle_protocol=torch.serialization.DEFAULT_PROTOCOL):
        super().__init__(filepath, pickle_module=pickle_module, pickle_protocol=pickle_protocol)
        self.filepath = filepath

    def on_end_epoch(self, model_state):
        super().on_end_training(model_state)
        self.save_checkpoint(model_state, overwrite_most_recent=True)


class Best(_Checkpointer):
    def __init__(self, filepath='model.{epoch:02d}-{val_loss:.2f}.pt',
            monitor='val_loss', mode='auto', period=1, min_delta=0, pickle_module=torch.serialization.pickle, pickle_protocol=torch.serialization.DEFAULT_PROTOCOL):
        super().__init__(filepath, pickle_module=pickle_module, pickle_protocol=pickle_protocol)
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.period = period
        self.epochs_since_last_save = 0

        if self.mode not in ['min', 'max']:
            if 'acc' in self.monitor:
                self.mode = 'max'
            else:
                self.mode = 'min'

        if self.mode == 'min':
            self.min_delta *= -1
            self.monitor_op = lambda x1, x2: (x1-self.min_delta) < x2
        elif self.mode == 'max':
            self.min_delta *= 1
            self.monitor_op = lambda x1, x2: (x1-self.min_delta) > x2

    def on_start(self, state):
        self.best = float('inf') if self.mode == 'min' else -float('inf')

    def on_end_epoch(self, model_state):

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            current = model_state['metrics'][self.monitor]

            if self.monitor_op(current, self.best):
                self.best = current
                self.save_checkpoint(model_state, overwrite_most_recent=True)


class Interval(_Checkpointer):
    def __init__(self, filepath='model.{epoch:02d}-{val_loss:.2f}.pt', period=1, pickle_module=torch.serialization.pickle, pickle_protocol=torch.serialization.DEFAULT_PROTOCOL):
        super().__init__(filepath, pickle_module=pickle_module, pickle_protocol=pickle_protocol)
        self.period = period
        self.epochs_since_last_save = 0

    def on_end_epoch(self, model_state):
        super().on_end_training(model_state)

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            self.save_checkpoint(model_state)



