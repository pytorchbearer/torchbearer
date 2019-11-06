from torchbearer import Callback
import torchbearer
from torch.distributions.beta import Beta
import random
import torch
import types


class ManifoldMixup(Callback):
    """

    Args:
        lam: Mixup lambda. If RANDOM, choose lambda from Beta(alpha, alpha). Else, lambda=lam
        alpha: Mixup alpha. Alpha used to sample lambda from Beta(alpha, alpha)
    """
    RANDOM = -10.0

    def __init__(self, alpha=1.0, lam=RANDOM):
        super(ManifoldMixup, self).__init__()
        self._layers = []
        self.mixup_layers = None
        self.alpha = alpha
        self.lam = lam
        self.distrib = Beta(self.alpha, self.alpha)
        self.layer_names = []
        self.depth = 0
        self.layer_filter = []
        self.layer_types = []

    def _sample_lam(self):
        if self.lam is self.RANDOM:
            if self.alpha > 0:
                lam = self.distrib.sample()
            else:
                lam = 1.0
        else:
            lam = self.lam
        return lam

    def _single_to_list(self, item):
        if not isinstance(item, list) or isinstance(item, tuple):
            item = [item, ]
        return item

    def for_layers(self, layers):
        """ Specify the layer names on which to apply manifold mixup.

        Args:
            layers (list or str): Layer names, eg ['conv1', 'fc1']

        Returns: self
        """
        layers = self._single_to_list(layers)
        self.mixup_layers = layers
        return self

    def with_layer_filter(self, layer_filter=()):
        """ Specify layer names to exclude from manifold mixup.

        Args:
            layer_filter (list or str): Layer filter, eg ['conv1', 'fc1']

        Returns: self
        """
        layer_filter = self._single_to_list(layer_filter)
        self.layer_filter = layer_filter
        return self

    def with_layer_type_filter(self, layer_types=()):
        """ Specify the layer types to exclude from manifold mixup

        Args:
            layer_types (list or str): Layer types, eg [nn.RelU]:

        Returns: self
        """
        layer_types = self._single_to_list(layer_types)
        self.layer_types = layer_types
        return self

    def at_depth(self, N):
        """ Specify the module depth at which to search for layers. Top level modules are at level 0. 
        Submodules of these are at level 1, etc. 
        To search all depths, set N=None.

        Args:
            N (int or None): Module depth

        Returns: self
        """
        self.depth = N
        return self

    def _wrap_layers(self, model, state):
        # Wrap the chosen layers with redefined forward that does mixup
        self._recursive_wrap(model, '', state, 0, self._layers)

    def on_start(self, state):
        super(ManifoldMixup, self).on_start(state)

        self._wrap_layers(state[torchbearer.MODEL], state)

        # if len(self._layers) == 0:
        #     raise Exception('Could not find desired layers. Not running manifold mixup.')

    def on_sample(self, state):
        # Choose layer to mixup and sample lambda
        lam = self._sample_lam()
        state[torchbearer.MIXUP_LAMBDA] = lam

        layer = random.choice(self._layers)
        layer.mixup()

        state[torchbearer.MIXUP_PERMUTATION] = torch.randperm(state[torchbearer.X].size(0))
        state[torchbearer.Y_TRUE] = (state[torchbearer.Y_TRUE], state[torchbearer.Y_TRUE][state[torchbearer.MIXUP_PERMUTATION]])

    def _wrap_layer_check(self, module, name, nname):
        # Check for exclusions

        name_check = self.mixup_layers is None or nname in self.mixup_layers

        filters = [
            # Filters
            any([f == nname for f in self.layer_filter]),
            any([isinstance(module, t) for t in self.layer_types]),
        ]
        return name_check and not any(filters)

    def _recursive_wrap(self, layer, pre_name, state, depth, filter=()):
        for name, module in layer.named_children():
            nname = pre_name + '_' + name if pre_name != '' else name

            def new_forward(old_forward):
                def new_forward_1(self, *args, **kwargs):
                    o = old_forward(*args, **kwargs)

                    if self.do_mixup:
                        o = mixup_inputs(o, state)

                    self.do_mixup = False
                    return o

                return new_forward_1

            if depth == self.depth or self.depth is None:
                self.layer_names.append(nname)

                if self._wrap_layer_check(module, name, nname):
                    self._layers.append(module)
                    nf = new_forward(module.forward)

                    bound_forward = types.MethodType(nf, module)
                    bound_mixup = types.MethodType(mixup, module)

                    module.__setattr__('forward', bound_forward)
                    module.__setattr__('mixup', bound_mixup)
                    module.__setattr__('state', state)
                    module.__setattr__('do_mixup', False)

            if self.depth is None or depth <= self.depth:
                if len(list(layer.named_children())) > 0:
                    self._recursive_wrap(module, nname, state, depth+1, filter)


def mixup(self):
    self.do_mixup = True


def mixup_inputs(x, state):
    index = state[torchbearer.MIXUP_PERMUTATION]
    lam = state[torchbearer.MIXUP_LAMBDA]
    mixed_x = lam * x + (1 - lam) * x[index,:]
    return mixed_x

