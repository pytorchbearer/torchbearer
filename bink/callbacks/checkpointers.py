from bink.callbacks.callbacks import Callback
import os


class _Checkpointer(Callback):
    def __init__(self, fileformat, save_weights_only):
        super().__init__()
        self.fileformat = fileformat
        self.save_weights_only = save_weights_only
        self.most_recent = None

    def save_checkpoint(self, model_state, overwrite_most_recent=False):
        state = {}
        state.update(model_state)
        state.update(model_state['metrics'])

        filepath = self.fileformat.format(**state)

        if self.most_recent is not None and overwrite_most_recent:
            os.rename(self.most_recent+'.pt', filepath+'.pt')
            os.rename(self.most_recent + '.bink', filepath + '.bink')

        state['self'].save(filepath + '.pt', filepath + '.bink', self.save_weights_only)
        self.most_recent = filepath


def ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}', monitor='val_loss', save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, min_delta=0):
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
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """
    if save_best_only:
        check = Best(filepath, monitor, save_weights_only, mode, period, min_delta)
    else:
        check = Interval(filepath, save_weights_only, period)

    return check


class MostRecent(_Checkpointer):
    def __init__(self, filepath='model.{epoch:02d}-{val_loss:.2f}', save_weights_only=False):
        super().__init__(filepath, save_weights_only)
        self.filepath = filepath
        self.save_weights_only = save_weights_only

    def on_end_training(self, model_state):
        super().on_end_training(model_state)
        self.save_checkpoint(model_state, overwrite_most_recent=True)


class Best(_Checkpointer):
    def __init__(self, filepath='model.{epoch:02d}-{val_loss:.2f}', monitor='val_loss', save_weights_only=False,
                 mode='auto', period=1, min_delta=0):
        super().__init__(filepath, save_weights_only)
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
            self.monitor_op = lambda x1, x2: x1 < x2
        elif self.mode == 'max':
            self.min_delta *= 1
            self.monitor_op = lambda x1, x2: x1 > x2

    def on_start(self, state):
        self.best = float('inf') if self.mode == 'min' else -float('inf')

    def on_end_training(self, model_state):

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            current = model_state['metrics'][self.monitor]

            if self.monitor_op(current, self.best):
                self.best = current
                self.save_checkpoint(model_state, overwrite_most_recent=True)


class Interval(_Checkpointer):
    def __init__(self, filepath='model.{epoch:02d}-{val_loss:.2f}', save_weights_only=False, period=1):
        super().__init__(filepath, save_weights_only)
        self.period = period
        self.epochs_since_last_save = 0

    def on_end_training(self, model_state):
        super().on_end_training(model_state)

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            self.save_checkpoint(model_state)



