import torchbearer
from torchbearer import Callback
import numpy as np
from torchbearer import cite

_bibtex = """
@inproceedings{smith2017cyclical,
  title={Cyclical learning rates for training neural networks},
  author={Smith, Leslie N},
  booktitle={2017 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={464--472},
  year={2017},
  organization={IEEE}
}
"""


@cite(_bibtex)
class CyclicLR(Callback):
    """ Learning rate finder that cyclicly varies the rate. Based off of the keras implementation referenced in the
    `paper <https://arxiv.org/abs/1506.01186>`_.

    Example: ::

        >>> import torch.nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import CyclicLR

        # Example Trial which does cyclic learning rate variation between 0.00001 and 0.0001 and back
        with period of 200 steps
        >>> lr_finder = CyclicLR(0.00001, 0.0001, step_size=100)
        >>> trial = Trial(model, callbacks=[lr_finder], metrics=['acc'])

    Args:
        base_lr (float / list): Float or list of floats for the base (min) learning rate for each optimiser parameter group
        max_lr (float / list):Float or list of floats for the max learning rate for each optimiser parameter group
        step_size (int / list): int or list of ints for the step size (half cyclic period) for each optimiser parameter group
        mode (str): One of (triangular, triangular2, exp_range) - the mode to use
        scale_fn (function): Scale function for learning rates over time. Default is defined by mode.
        scale_mode (str): One of (cycle, iterations). Argument passed to the scale function each step
        gamma (float): Scaling factor for exp_range mode

    State Requirements:
        - :attr:`torchbearer.state.OPTIMIZER`: State should have the current optimiser stored
    """
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000, mode='triangular', scale_fn=None, scale_mode='cycle', gamma=1.):
        super(CyclicLR, self).__init__()

        if not isinstance(base_lr, list):
            base_lr = [base_lr]
        if not isinstance(max_lr, list):
            max_lr = [max_lr]
        if not isinstance(step_size, list):
            step_size = [step_size]

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.iterations = 0
        self.optim = None
        self.mode = mode
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode

        if scale_fn is None:
            if mode == 'triangular':
                self.scale_fn = lambda x: 1.0
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**x
                self.scale_mode = 'iterations'

    def on_start(self, state):
        super(CyclicLR, self).on_start(state)
        self.optim = state[torchbearer.OPTIMIZER]
        self.iterations = 0

        if len(self.optim.param_groups) > 1:
            if len(self.base_lr) == 1:
                self.base_lr = self.base_lr * len(self.optim.param_groups)
            if len(self.max_lr) == 1:
                self.max_lr = self.max_lr * len(self.optim.param_groups)
            if len(self.step_size) == 1:
                self.step_size = self.step_size * len(self.optim.param_groups)

    def on_sample(self, state):
        super(CyclicLR, self).on_sample(state)
        self.update_lrs()

    def on_step_training(self, state):
        super(CyclicLR, self).on_step_training(state)
        self.iterations = self.iterations + 1

    def update_lrs(self):
        i = 0
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.next_lr(self.iterations, i)
            i += 1

    def next_lr(self, epoch_count, group_id):
        epoch_count = float(epoch_count)
        cycle = np.floor(1.0 + epoch_count / (2.0*self.step_size[group_id]))
        x = np.abs(epoch_count / self.step_size[group_id] - 2.0*cycle + 1.0)
        new_lr = self.base_lr[group_id] + (self.max_lr[group_id]-self.base_lr[group_id])*np.maximum(0.0, (1.0-x))

        scale_param = cycle
        if self.scale_mode != 'cycle':
            scale_param = self.iterations

        return new_lr*self.scale_fn(scale_param)
