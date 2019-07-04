import torch

import torchbearer
from torchbearer import cite
from torchbearer.callbacks import Callback, only_if, once_per_epoch


bibtex = """
@article{inoue2018data,
  title={Data augmentation by pairing samples for images classification},
  author={Inoue, Hiroshi},
  journal={arXiv preprint arXiv:1801.02929},
  year={2018}
}
"""


@cite(bibtex)
class SamplePairing(Callback):
    """Perform SamplePairing on the model inputs. This is the process of averaging each image with another random image
    without changing the targets. The key here is to use the policy function to only do this some of the time.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import SamplePairing

        # Example Trial which does Sample Pairing regularisation with the policy from the paper
        >>> pairing = SamplePairing()
        >>> trial = Trial(None, criterion=Mixup.loss, callbacks=[pairing], metrics=['acc'])

    Args:
        policy: A function of state which returns True if the current batch should be paired.
    """

    @staticmethod
    def default_policy(start_epoch, end_epoch, on_epochs, off_epochs):
        """Return a policy which performs sample pairing according to the process defined in the paper.

        Args:
            start_epoch (int): Epoch to start pairing on
            end_epoch (int): Epoch to end pairing on (and begin fine-tuning)
            on_epochs (int): Number of epochs to run sample pairing for before a break
            off_epochs (int): Number of epochs to break for

        Returns:
            A policy function
        """
        cache = {'tick': 0, 'on': False}

        @once_per_epoch
        def compute(state):
            if start_epoch <= state[torchbearer.EPOCH] < end_epoch:
                if cache['tick'] < on_epochs:
                    cache['on'] = True
                else:
                    cache['on'] = False

                cache['tick'] = cache['tick'] + 1
                if cache['tick'] == (on_epochs + off_epochs):
                    cache['tick'] = 0
                    cache['on'] = False
            else:
                cache['on'] = False

        def policy(state):
            compute(state)
            return cache['on']

        return policy

    def __init__(self, policy=None):
        self.on_sample = only_if(SamplePairing.default_policy(100, 800, 8, 2) if policy is None else policy)(self.on_sample)

    def on_sample(self, state):
        permutation = torch.randperm(state[torchbearer.X].size(0))
        state[torchbearer.INPUT] = (state[torchbearer.INPUT] + state[torchbearer.INPUT][permutation]) / 2.
