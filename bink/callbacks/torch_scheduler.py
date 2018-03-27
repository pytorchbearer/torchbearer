from bink.callbacks import Callback

import torch


class TorchScheduler(Callback):
    def __init__(self, scheduler, monitor=None):
        self._scheduler = scheduler
        self._monitor = monitor

    def on_start_training(self, state):
        if self._monitor is None:
            self._scheduler.step()

    def on_end_training(self, state):
        if self._monitor is not None:
            self._scheduler.step(state['final_metrics'][self._monitor])


class LambdaLR(TorchScheduler):
    """
    See:
        `PyTorch LambdaLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.LambdaLR>`_
    """
    def __init__(self, *args, **kwargs):
        super(LambdaLR, self).__init__(torch.optim.lr_scheduler.LambdaLR(*args, **kwargs))


class StepLR(TorchScheduler):
    """
    See:
        `PyTorch StepLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.StepLR>`_
    """
    def __init__(self, *args, **kwargs):
        super(StepLR, self).__init__(torch.optim.lr_scheduler.StepLR(*args, **kwargs))


class MultiStepLR(TorchScheduler):
    """
    See:
        `PyTorch MultiStepLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.MultiStepLR>`_
    """
    def __init__(self, *args, **kwargs):
        super(MultiStepLR, self).__init__(torch.optim.lr_scheduler.MultiStepLR(*args, **kwargs))


class ExponentialLR(TorchScheduler):
    """
    See:
        `PyTorch ExponentialLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ExponentialLR>`_
    """
    def __init__(self, *args, **kwargs):
        super(ExponentialLR, self).__init__(torch.optim.lr_scheduler.ExponentialLR(*args, **kwargs))


class CosineAnnealingLR(TorchScheduler):
    """
    See:
        `PyTorch CosineAnnealingLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR>`_
    """
    def __init__(self, *args, **kwargs):
        super(CosineAnnealingLR, self).__init__(torch.optim.lr_scheduler.CosineAnnealingLR(*args, **kwargs))


class ReduceLROnPlateau(TorchScheduler):
    """
    Args:
        monitor (string): The quantity to monitor. (Default value = 'val_loss')
    See:
        `PyTorch ReduceLROnPlateau <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau>`_
    """
    def __init__(self, *args, monitor='val_loss', **kwargs):
        super(ReduceLROnPlateau, self).__init__(torch.optim.lr_scheduler.ReduceLROnPlateau(*args, **kwargs), monitor=monitor)