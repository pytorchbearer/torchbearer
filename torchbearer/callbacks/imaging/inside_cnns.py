import torch
import torch.nn as nn
import torch.optim as optim
import torchbearer
from torchbearer.callbacks.decorators import once_per_epoch
from . import imaging

_inside_cnns = """
@article{simonyan2013deep,
  title={Deep inside convolutional networks: Visualising image classification models and saliency maps},
  author={Simonyan, Karen and Vedaldi, Andrea and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1312.6034},
  year={2013}
}
"""

RANDOM = -10
""" Flag that when passed as the target chosses a random target"""


class _CAMWrapper(nn.Module):
    def __init__(self, input_size, base_model, transform=None):
        super(_CAMWrapper, self).__init__()
        self.base_model = base_model
        input_image = torch.zeros(input_size)

        self.input_image = nn.Parameter(input_image, requires_grad=True)

        self.transform = (lambda x: x) if transform is None else transform

    def forward(self, _, state):
        try:
            return self.base_model(self.transform(self.input_image.sigmoid()).unsqueeze(0), state)
        except TypeError:
            return self.base_model(self.transform(self.input_image.sigmoid()).unsqueeze(0))


def _cam_loss(key, targets_hot, decay):
    def loss(state):
        img = state[torchbearer.MODEL].input_image
        return - torch.masked_select(state[key], targets_hot).sum() + decay * img.pow(2).sum()
    return loss


@torchbearer.cite(_inside_cnns)
class ClassAppearanceModel(imaging.ImagingCallback):
    """The :class:`.ClassAppearanceModel` callback implements Figure 1 from
    `Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps <https://arxiv.org/abs/1312.6034>`_.
    This is a simple gradient ascent on an image (initialised to zero) with a sum-squares regularizer. Internally this
    creates a new :class:`.Trial` instance which then performs the optimization.

    Args:
        nclasses (int): The number of output classes
        input_size (tuple): The size to use for the input image
        optimizer_factory: A function of parameters which returns an optimizer to use
        logit_key (StateKey): :class:`.StateKey` storing the class logits
        target (int): Target class for the optimisation or RANDOM
        steps (int): Number of optimisation steps to take
        decay (float): Lambda for the L2 decay on the image
        verbose (int): Verbosity level to pass to the internal :class:`.Trial` instance
        transform (callable, optional): A function/transform that  takes in a Tensor and returns a transformed version.
            This will be applied to the image before it is sent to output

    """
    def __init__(self, nclasses, input_size, optimizer_factory=lambda params: optim.Adam(params, lr=0.5), steps=256,
                 logit_key=torchbearer.PREDICTION, target=RANDOM, decay=0.01, verbose=0, in_transform=None, transform=None):
        super(ClassAppearanceModel, self).__init__(transform=transform)

        self.nclasses = nclasses
        self.input_size = input_size
        self.optimizer_factory = optimizer_factory
        self.logit_key = logit_key
        self.target = target
        self.steps = steps
        self.decay = decay
        self.verbose = verbose
        self.in_transform = in_transform

        self._target_keys = []

    def target_to_key(self, key):
        self._target_keys.append(key)
        return self

    def _targets_hot(self, state):
        targets = torch.randint(high=self.nclasses, size=(1, 1)).long().to(state[torchbearer.DEVICE])
        if self.target is not RANDOM:
            targets[0][0] = self.target
        for key in self._target_keys:
            state[key] = targets
        targets_hot = torch.zeros(1, self.nclasses).to(state[torchbearer.DEVICE])
        targets_hot.scatter_(1, targets, 1)
        targets_hot = targets_hot.ge(0.5)
        return targets_hot

    @torchbearer.enable_grad()
    @once_per_epoch
    def on_batch(self, state):
        training = state[torchbearer.MODEL].training
        state[torchbearer.MODEL].eval()

        targets_hot = self._targets_hot(state)

        key = self.logit_key

        @torchbearer.callbacks.on_sample
        def make_eval(_):
            state[torchbearer.MODEL].eval()

        model = _CAMWrapper(self.input_size, state[torchbearer.MODEL], transform=self.in_transform)
        trial = torchbearer.Trial(model, self.optimizer_factory(filter(lambda p: p.requires_grad, [model.input_image])),
                                  _cam_loss(key, targets_hot, self.decay), callbacks=[make_eval])
        trial.for_train_steps(self.steps).to(state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])
        trial.run(verbose=self.verbose)

        if training:
            state[torchbearer.MODEL].train()

        return model.input_image.sigmoid()
