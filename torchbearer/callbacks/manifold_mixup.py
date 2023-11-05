from torchbearer import Callback
import torchbearer
from torch.distributions.beta import Beta
import random
import torch
import types
from torchbearer import cite


manifold_mixup = """
@inproceedings{verma2019manifold,
  title={Manifold Mixup: Better Representations by Interpolating Hidden States},
  author={Verma, Vikas and Lamb, Alex and Beckham, Christopher and Najafi, Amir and Mitliagkas, Ioannis and Lopez-Paz, David and Bengio, Yoshua},
  booktitle={International Conference on Machine Learning},
  pages={6438--6447},
  year={2019}
}
"""


@cite(manifold_mixup)
class ManifoldMixup(Callback):
    """ Performs manifold mixup on desired layers. Requires use of :meth:`MixupInputs.loss`, otherwise lambdas can be found in
    state under :attr:`.MIXUP_LAMBDA`. Model targets will be a tuple containing the original target and permuted target.

    Args:
        lam: Mixup lambda. If RANDOM, choose lambda from Beta(alpha, alpha). Else, lambda=lam
        alpha: Mixup alpha. Alpha used to sample lambda from Beta(alpha, alpha)
    """
    RANDOM = -10.0

    def __init__(self, alpha=1.0, lam=RANDOM):
        super(ManifoldMixup, self).__init__()
        self._layers = []
        self._mixup_layers = None
        self.alpha = alpha
        self.lam = lam
        if alpha > 0:
            self.distrib = Beta(self.alpha, self.alpha)
        self.layer_names = []
        self.depth = 0
        self._layer_filter = []
        self._layer_types = []

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
        self._mixup_layers = layers
        return self

    def with_layer_filter(self, layer_filter=()):
        """ Specify layer names to exclude from manifold mixup.

        Args:
            layer_filter (list or str): Layer filter, eg ['conv1', 'fc1']

        Returns: self
        """
        layer_filter = self._single_to_list(layer_filter)
        self._layer_filter = layer_filter
        return self

    def with_layer_type_filter(self, layer_types=()):
        """ Specify the layer types to exclude from manifold mixup

        Args:
            layer_types (list or str): Layer types, eg [nn.RelU]:

        Returns: self
        """
        layer_types = self._single_to_list(layer_types)
        self._layer_types = layer_types
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
        self._recursive_wrap(model, '', state, 0)

    def on_start(self, state):
        super(ManifoldMixup, self).on_start(state)
        self._wrap_layers(state[torchbearer.MODEL], state)

    def on_sample(self, state):
        lam = self._sample_lam()
        state[torchbearer.MIXUP_LAMBDA] = lam

        layer = random.choice(self._layers)
        layer.mixup()

        state[torchbearer.MIXUP_PERMUTATION] = torch.randperm(state[torchbearer.X].size(0))
        state[torchbearer.Y_TRUE] = (state[torchbearer.Y_TRUE], state[torchbearer.Y_TRUE][state[torchbearer.MIXUP_PERMUTATION]])
        
    def _wrap_layer_check(self, module, name, nname):
        # Check for exclusions
        name_check = self._mixup_layers is None or nname in self._mixup_layers

        filters = [
            any([f == nname for f in self._layer_filter]),
            any([isinstance(module, t) for t in self._layer_types]),
        ]
        return name_check and not any(filters)
    
    def get_selected_layers(self, model):
        layer_names = []
        return self._recursive_name_seach(layer_names, model, '', 0)

    def _recursive_name_seach(self, layer_names, layer, pre_name, depth):
        for name, module in layer.named_children():
            nname = pre_name + '_' + name if pre_name != '' else name
            if depth == self.depth or self.depth is None:
                if self._wrap_layer_check(module, name, nname):
                    layer_names.append(nname)
                
            if self.depth is None or depth <= self.depth:
                if len(list(layer.named_children())) > 0:
                    self._recursive_name_seach(layer_names, module, nname, depth+1)
        return layer_names

    def _recursive_wrap(self, layer, pre_name, state, depth):
        for name, module in layer.named_children():
            nname = pre_name + '_' + name if pre_name != '' else name

            def new_forward(old_forward):
                def new_forward_1(self, *args, **kwargs):
                    o = old_forward(*args, **kwargs)
                    
                    if self.do_mixup and self.training:
                        o = _mixup_inputs(o, state)

                    self.do_mixup = False
                    return o

                return new_forward_1

            if depth == self.depth or self.depth is None:
                self.layer_names.append(nname)

                if self._wrap_layer_check(module, name, nname):
                    self._layers.append(module)
                    nf = new_forward(module.forward)

                    bound_forward = types.MethodType(nf, module)
                    bound_mixup = types.MethodType(_mixup, module)

                    module.__setattr__('forward', bound_forward)
                    module.__setattr__('mixup', bound_mixup)
                    module.__setattr__('state', state)
                    module.__setattr__('do_mixup', False)

            if self.depth is None or depth <= self.depth:
                if len(list(layer.named_children())) > 0:
                    self._recursive_wrap(module, nname, state, depth+1)


def _mixup(self):
    self.do_mixup = True


def _mixup_inputs(x, state):
    index = state[torchbearer.MIXUP_PERMUTATION]
    lam = state[torchbearer.MIXUP_LAMBDA]
    mixed_x = lam * x + (1 - lam) * x[index,:]
    return mixed_x

