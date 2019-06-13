import torchbearer

from torchbearer.callbacks import Callback

import torch


class TorchScheduler(Callback):
    def __init__(self, scheduler_builder, monitor=None, step_on_batch=False):
        self._scheduler_builder = scheduler_builder
        self._monitor = monitor
        self._scheduler = None
        self._step_on_batch = step_on_batch

    def on_start(self, state):
        self._scheduler = self._scheduler_builder(state[torchbearer.OPTIMIZER])

    def on_sample(self, state):
        if self._step_on_batch and self._monitor is None:
            self._scheduler.step()

    def on_step_training(self, state):
        if self._step_on_batch and self._monitor is not None:
            self._scheduler.step(state[torchbearer.METRICS][self._monitor])

    def on_start_training(self, state):
        if not self._step_on_batch and self._monitor is None:
            self._scheduler.step(epoch=state[torchbearer.EPOCH])

    def on_end_epoch(self, state):
        if not self._step_on_batch and self._monitor is not None:
            self._scheduler.step(state[torchbearer.METRICS][self._monitor], epoch=state[torchbearer.EPOCH])


class LambdaLR(TorchScheduler):
    """
    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import LambdaLR

        # Example Trial which performs the two learning rate lambdas from the PyTorch docs
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(lr_lambda=[lambda1, lambda2])
        >>> trial = Trial(None, callbacks=[scheduler], metrics=['loss'], verbose=2).for_steps(10).run(1)

    Args:
        step_on_batch (bool): If True, step will be called on each training iteration rather than on each epoch

    See:
        `PyTorch LambdaLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.LambdaLR>`_
    """
    def __init__(self, lr_lambda, last_epoch=-1, step_on_batch=False):
        super(LambdaLR, self).__init__(lambda opt:
                                       torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=last_epoch),
                                       step_on_batch=step_on_batch)


class StepLR(TorchScheduler):
    """
    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import StepLR

        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> scheduler = StepLR(step_size=30, gamma=0.1)
        >>> trial = Trial(None, callbacks=[scheduler], metrics=['loss'], verbose=2).for_steps(10).run(1)

    Args:
        step_on_batch (bool): If True, step will be called on each training iteration rather than on each epoch

    See:
        `PyTorch StepLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.StepLR>`_
    """
    def __init__(self, step_size, gamma=0.1, last_epoch=-1, step_on_batch=False):
        super(StepLR, self).__init__(lambda opt:
                                     torch.optim.lr_scheduler.StepLR(opt, step_size, gamma=gamma,
                                                                     last_epoch=last_epoch),
                                     step_on_batch=step_on_batch)


class MultiStepLR(TorchScheduler):
    """
    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import MultiStepLR

        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(milestones=[30,80], gamma=0.1)
        >>> trial = Trial(None, callbacks=[scheduler], metrics=['loss'], verbose=2).for_steps(10).run(1)

    Args:
        step_on_batch (bool): If True, step will be called on each training iteration rather than on each epoch

    See:
        `PyTorch MultiStepLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.MultiStepLR>`_
    """
    def __init__(self, milestones, gamma=0.1, last_epoch=-1, step_on_batch=False):
        super(MultiStepLR, self).__init__(lambda opt:
                                          torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=gamma,
                                                                               last_epoch=last_epoch),
                                          step_on_batch=step_on_batch)


class ExponentialLR(TorchScheduler):
    """
    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import ExponentialLR

        >>> # Example scheduler which multiplies the learning rate by 0.1 every epoch
        >>> scheduler = ExponentialLR(gamma=0.1)
        >>> trial = Trial(None, callbacks=[scheduler], metrics=['loss'], verbose=2).for_steps(10).run(1)

    Args:
        step_on_batch (bool): If True, step will be called on each training iteration rather than on each epoch

    See:
        `PyTorch ExponentialLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ExponentialLR>`_
    """
    def __init__(self, gamma, last_epoch=-1, step_on_batch=False):
        super(ExponentialLR, self).__init__(lambda opt:
                                            torch.optim.lr_scheduler.ExponentialLR(opt, gamma, last_epoch=last_epoch),
                                            step_on_batch=step_on_batch)


class CosineAnnealingLR(TorchScheduler):
    """
    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import CosineAnnealingLR

        >>> # Example scheduler which uses cosine learning rate annealing - see PyTorch docs
        >>> scheduler = MultiStepLR(milestones=[30,80], gamma=0.1)
        >>> trial = Trial(None, callbacks=[scheduler], metrics=['loss'], verbose=2).for_steps(10).run(1)

    Args:
        step_on_batch (bool): If True, step will be called on each training iteration rather than on each epoch

    See:
        `PyTorch CosineAnnealingLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR>`_
    """
    def __init__(self, T_max, eta_min=0, last_epoch=-1, step_on_batch=False):
        super(CosineAnnealingLR, self).__init__(lambda opt:
                                                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max, eta_min=eta_min,
                                                                                           last_epoch=last_epoch),
                                                step_on_batch=step_on_batch)


class ReduceLROnPlateau(TorchScheduler):
    """
    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import ReduceLROnPlateau

        >>> # Example scheduler which multiplies the learning rate by 10 on plateaus of 5 epochs without significant
        >>> # validation loss decrease
        >>> scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
        >>> trial = Trial(None, callbacks=[scheduler], metrics=['loss'], verbose=2).for_steps(10).for_val_steps(10).run(1)

    Args:
        monitor (str): The name of the quantity in metrics to monitor. (Default value = 'val_loss')
        step_on_batch (bool): If True, step will be called on each training iteration rather than on each epoch

    See:
        `PyTorch ReduceLROnPlateau <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau>`_
    """
    def __init__(self,  monitor='val_loss', mode='min', factor=0.1, patience=10, verbose=False, threshold=1e-4,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, step_on_batch=False):
        super(ReduceLROnPlateau, self).__init__(lambda opt:
                                                torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                    opt, mode=mode, factor=factor, patience=patience, verbose=verbose,
                                                    threshold=threshold, threshold_mode=threshold_mode,
                                                    cooldown=cooldown, min_lr=min_lr, eps=eps), monitor=monitor,
                                                step_on_batch=step_on_batch)
