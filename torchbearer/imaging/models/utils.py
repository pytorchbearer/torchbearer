import torch.nn
import torchbearer


LAYER_DICT = torchbearer.state_key('layer_dict')

_relu_classes = [torch.nn.ReLU, torch.nn.ReLU6, torch.nn.LeakyReLU, torch.nn.RReLU, torch.nn.PReLU]


class Storer(torch.nn.Module):
    def __init__(self, save_state, name):
        super().__init__()
        self.save_state = save_state
        self.name = name

    def forward(self, x):
        self.save_state[self.name] = x
        return x


def storer(state, name, x):
    try:
        state[LAYER_DICT][name] = x
    except:
        state[LAYER_DICT] = {name: x}
    return x


def basemodel(model):
    class BaseModel(model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            relus_to_not_inplace(self)
            self.__name__ = model.__name__
            self.__qualname__ = model.__qualname__

        def _get_name(self):
            return self.__name__

        def forward(self, x, state):
            super().forward(x, state)

    return BaseModel


def relus_to_not_inplace(model):
    for m in model.modules():
        if type(m) in _relu_classes:
            m.inplace = False
