import torchbearer
from torchbearer import cite
from torchbearer.callbacks import Callback

import torch.nn.init as init

__kaiming__ = """
@inproceedings{he2015delving,
  title={Delving deep into rectifiers: Surpassing human-level performance on imagenet classification},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={1026--1034},
  year={2015}
}"""

__xavier__ = """
@inproceedings{glorot2010understanding,
  title={Understanding the difficulty of training deep feedforward neural networks},
  author={Glorot, Xavier and Bengio, Yoshua},
  booktitle={Proceedings of the thirteenth international conference on artificial intelligence and statistics},
  pages={249--256},
  year={2010}
}
"""

__lsuv__ = """
@article{mishkin2015all,
  title={All you need is a good init},
  author={Mishkin, Dmytro and Matas, Jiri},
  journal={arXiv preprint arXiv:1511.06422},
  year={2015}
}
"""


class WeightInit(Callback):
    """Base class for weight initialisations. Performs the provided function for each module when on_init is
    called.

    Args:
        initialiser (lambda): a function which initialises an nn.Module **inplace**
        modules (Iterable[nn.Module] or nn.Module, optional): an iterable of nn.Modules or a
            single nn.Module that will have weights initialised, otherwise this is retrieved from the model
        targets (list[String]): A list of lookup strings to match which modules will be initialised

    State Requirements:
        - :attr:`torchbearer.state.MODEL`: Model should have the `modules` method if modules is None
    """
    def __init__(self, initialiser=lambda module: module, modules=None, targets=['Conv', 'Linear', 'Bilinear']):
        self.initialiser = initialiser
        self.modules = modules
        self.targets = targets

    def on_init(self, state):
        if self.modules is None:
            self.modules = state[torchbearer.MODEL].modules()

        for m in self.modules:
            if len(list(filter(lambda target: target in m.__class__.__name__, self.targets))) > 0:
                self.initialiser(m)


@cite(__lsuv__)
class LsuvInit(Callback):
    """Layer-sequential unit-variance (LSUV) initialization as described in
    `All you need is a good init <https://arxiv.org/abs/1511.06422>`_ and
    modified from the code by  `ducha-aiki <https://github.com/ducha-aiki/LSUV-pytorch>`__.
    To be consistent with the paper, LsuvInit should be preceeded by a ZeroBias init on the Linear and Conv layers.


    Example: ::

        >>> import torch
        >>> import torch.nn as nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks.init import LsuvInit

        # 100 random data points
        >>> data = torch.rand(100, 3, 5, 5)
        >>> example_batch = data[:3]
        >>> lsuv = LsuvInit(example_batch)

        # Model and trail using lsuv init for some random data
        >>> model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU())
        >>> trial = Trial(model, callbacks=[lsuv]).with_train_data(data, data+5)

    Args:
        data_item (torch.Tensor): A representative data item to put through the model
        weight_lambda (lambda): A function that takes a module and returns the weight attribute. If none defaults to 
            module.weight.
        needed_std: See `paper <https://arxiv.org/abs/1511.06422>`__, where needed_std is always 1.0
        std_tol: See `paper <https://arxiv.org/abs/1511.06422>`__, Tol_{var}
        max_attempts: See `paper <https://arxiv.org/abs/1511.06422>`__, T_{max}
        do_orthonorm: See `paper <https://arxiv.org/abs/1511.06422>`__, first pre-initialise with orthonormal matricies

    State Requirements:
        - :attr:`torchbearer.state.MODEL`: Model should have the `modules` method if modules is None
    """
    def __init__(self, data_item, weight_lambda=None, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=True):
        from torchbearer.callbacks.lsuv import LSUV
        self.lsuv_init = LSUV
        self.data = data_item
        self.needed_std = needed_std
        self.std_tol = std_tol
        self.max_attempts = max_attempts
        self.do_arthonorm = do_orthonorm
        self.weight_lambda = weight_lambda

    def on_init(self, state):
        lsuv = self.lsuv_init()
        state[torchbearer.MODEL] = lsuv.init_model(state[torchbearer.MODEL], self.data, self.weight_lambda, self.needed_std,
                                                  self.std_tol, self.max_attempts, self.do_arthonorm)


@cite(__kaiming__)
class KaimingNormal(WeightInit):
    """Kaiming Normal weight initialisation. Uses ``torch.nn.init.kaiming_normal_`` on the ``weight`` attribute of the
    filtered modules.

    Example: ::

        >>> import torch
        >>> import torch.nn as nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks.init import KaimingNormal

        # 100 random data points
        >>> data = torch.rand(100, 3, 5, 5)
        >>> example_batch = data[:3]
        >>> initialiser = KaimingNormal()

        # Model and trail using kaiming init for some random data
        >>> model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU())
        >>> trial = Trial(model, callbacks=[initialiser]).with_train_data(data, data+5)

    Args:
        a (int): See `PyTorch kaiming_uniform_ <https://pytorch.org/docs/stable/nn.html#torch.nn.init.kaiming_uniform_>`_
        mode (str): See `PyTorch kaiming_uniform_`_
        nonlinearity (str): See `PyTorch kaiming_uniform_`_
        modules (Iterable[nn.Module] or nn.Module, optional): an iterable of nn.Modules or a
            single nn.Module that will have weights initialised, otherwise this is retrieved from the model
        targets (list[String]): A list of lookup strings to match which modules will be initialised

    See:
        `PyTorch kaiming_normal_`_
    """
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu', modules=None,
                 targets=['Conv', 'Linear', 'Bilinear']):
        def initialiser(module):
            init.kaiming_normal_(module.weight.data, a=a, mode=mode, nonlinearity=nonlinearity)

        super(KaimingNormal, self).__init__(initialiser, modules=modules, targets=targets)


@cite(__kaiming__)
class KaimingUniform(WeightInit):
    """Kaiming Uniform weight initialisation. Uses ``torch.nn.init.kaiming_uniform_`` on the ``weight`` attribute of the
    filtered modules.

    Example: ::

        >>> import torch
        >>> import torch.nn as nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks.init import KaimingUniform

        # 100 random data points
        >>> data = torch.rand(100, 3, 5, 5)
        >>> example_batch = data[:3]
        >>> initialiser = KaimingUniform()

        # Model and trail using kaiming init for some random data
        >>> model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU())
        >>> trial = Trial(model, callbacks=[initialiser]).with_train_data(data, data+5)

    Args:
        a (int): See `PyTorch kaiming_uniform_ <https://pytorch.org/docs/stable/nn.html#torch.nn.init.kaiming_uniform_>`_
        mode (str): See `PyTorch kaiming_uniform_`_
        nonlinearity (str): See `PyTorch kaiming_uniform_`_
        modules (Iterable[nn.Module] or nn.Module, optional): an iterable of nn.Modules or a
            single nn.Module that will have weights initialised, otherwise this is retrieved from the model
        targets (list[String]): A list of lookup strings to match which modules will be initialised

    See:
        `PyTorch kaiming_uniform_`_
    """
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu', modules=None,
                 targets=['Conv', 'Linear', 'Bilinear']):
        def initialiser(module):
            init.kaiming_uniform_(module.weight.data, a=a, mode=mode, nonlinearity=nonlinearity)

        super(KaimingUniform, self).__init__(initialiser, modules=modules, targets=targets)


@cite(__xavier__)
class XavierNormal(WeightInit):
    """Xavier Normal weight initialisation. Uses ``torch.nn.init.xavier_normal_`` on the ``weight`` attribute of the
    filtered modules.

    Example: ::

        >>> import torch
        >>> import torch.nn as nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks.init import XavierNormal

        # 100 random data points
        >>> data = torch.rand(100, 3, 5, 5)
        >>> example_batch = data[:3]
        >>> initialiser = XavierNormal()

        # Model and trail using Xavier init for some random data
        >>> model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU())
        >>> trial = Trial(model, callbacks=[initialiser]).with_train_data(data, data+5)

    Args:
        gain (int): See `PyTorch xavier_normal_ <https://pytorch.org/docs/stable/nn.html#torch.nn.init.xavier_normal_>`_
        modules (Iterable[nn.Module] or nn.Module, optional): an iterable of nn.Modules or a
            single nn.Module that will have weights initialised, otherwise this is retrieved from the model
        targets (list[String]): A list of lookup strings to match which modules will be initialised

    See:
        `PyTorch xavier_normal_`_
    """
    def __init__(self, gain=1, modules=None, targets=['Conv', 'Linear', 'Bilinear']):
        def initialiser(module):
            init.xavier_normal_(module.weight.data, gain=gain)

        super(XavierNormal, self).__init__(initialiser, modules=modules, targets=targets)


@cite(__xavier__)
class XavierUniform(WeightInit):
    """Xavier Uniform weight initialisation. Uses ``torch.nn.init.xavier_uniform_`` on the ``weight`` attribute of the
    filtered modules.

    Example: ::

        >>> import torch
        >>> import torch.nn as nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks.init import XavierUniform

        # 100 random data points
        >>> data = torch.rand(100, 3, 5, 5)
        >>> example_batch = data[:3]
        >>> initialiser = XavierUniform()

        # Model and trail using Xavier init for some random data
        >>> model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU())
        >>> trial = Trial(model, callbacks=[initialiser]).with_train_data(data, data+5)

    Args:
        gain (int): See `PyTorch xavier_normal_ <https://pytorch.org/docs/stable/nn.html#torch.nn.init.xavier_normal_>`_
        modules (Iterable[nn.Module] or nn.Module, optional): an iterable of nn.Modules or a
            single nn.Module that will have weights initialised, otherwise this is retrieved from the model
        targets (list[String]): A list of lookup strings to match which modules will be initialised

    See:
        `PyTorch xavier_uniform_`_
    """
    def __init__(self, gain=1, modules=None, targets=['Conv', 'Linear', 'Bilinear']):
        def initialiser(module):
            init.xavier_uniform_(module.weight.data, gain=gain)

        super(XavierUniform, self).__init__(initialiser, modules=modules, targets=targets)


class ZeroBias(WeightInit):
    """Zero initialisation for the ``bias`` attributes of filtered modules. This is recommended for use in conjunction
    with weight initialisation schemes.

    Example: ::

        >>> import torch
        >>> import torch.nn as nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks.init import ZeroBias

        # 100 random data points
        >>> data = torch.rand(100, 3, 5, 5)
        >>> example_batch = data[:3]
        >>> initialiser = ZeroBias()

        # Model and trail using zero bias init for some random data
        >>> model = nn.Sequential(nn.Conv2d(3, 1, 3), nn.ReLU())
        >>> trial = Trial(model, callbacks=[initialiser]).with_train_data(data, data+5)

    Args:
        modules (Iterable[nn.Module] or nn.Module, optional): an iterable of nn.Modules or a
            single nn.Module that will have weights initialised, otherwise this is retrieved from the model
        targets (list[String]): A list of lookup strings to match which modules will be initialised
    """
    def __init__(self, modules=None, targets=['Conv', 'Linear', 'Bilinear']):
        def initialiser(module):
            module.bias.data.zero_()

        super(ZeroBias, self).__init__(initialiser, modules=modules, targets=targets)
