import torchbearer
from torchbearer import Callback


class UnpackState(Callback):
    """Callback that unpacks a number of items from the state dictionary and passes it to the model forward as a separate
    dictionary (under the same keys).

    Note that torchbearer.X is always passed in the dictionary, if given as a key or not.

    Note that, if output_to_state is set then when the model outputs a dictionary, the main
    torchbearer state will be updated with this dictionary. So, if model outputs are
    {torchbearer.Y_PRED: 1, SOME_KEY: 10} then the main torchbearer state will look like:
    {..., torchbearer.Y_PRED: 1, SOME_KEY: 10, ...} after the forward pass. If Y_PRED is not a key in the model output
    dict or the output is not a dict then main state will look like: {..., torchbearer.Y_PRED: (model_output), ...}

    This is useful when using a DataParallel model where DataParallel cannot pass the main torchbearer state dictionary
    directly.

    Example: ::
        >>> import torch
        >>> from torch.nn import Module, DataParallel
        >>> import torchbearer
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import UnpackState

        >>> class TestModel(Module):
        ...     def forward(self, x):
        ...         print(x)
        ...         return x

        >>> A_KEY = torchbearer.state_key('a_key')
        >>> unpacker = UnpackState(keys=[torchbearer.X, A_KEY])
        >>> t = Trial(TestModel(), callbacks=[unpacker])
        >>> t.state[A_KEY] = 'test'
        >>> t.with_train_data(torch.ones(10, 1), torch.ones(10, 1), batch_size=2, steps=1).run()
        {x: tensor([[1.],
                [1.]]), a_key: 'test'}
        [((1, None), {})]


    Args:
        keys (list / tuple): List of keys to unpack from state and pass to the model on forward
        output_to_state (bool): If True and torchbearer.Y_PRED in model_outputs, main torchbearer state will be updated
        (state.update(model_outputs)). If False then model outputs will be stored in the main state under
        torchbeaerer.Y_PRED.
    """
    def __init__(self, keys=None, output_to_state=True):
        super(UnpackState, self).__init__()
        self.output_to_state = output_to_state
        self.keys = keys if keys is not None else [torchbearer.X]
        if torchbearer.X not in self.keys:
            self.keys.insert(0, torchbearer.X)

    def on_sample(self, state):
        if self.keys != [torchbearer.X]:
            state[torchbearer.X] = {k: state[k] for k in self.keys}

    def on_sample_validation(self, state):
        self.on_sample(state)

    def on_forward(self, state):
        if self.output_to_state and isinstance(state[torchbearer.Y_PRED], dict):
            outputs = state[torchbearer.Y_PRED]
            state.update(outputs)
            state[torchbearer.Y_PRED] = outputs[torchbearer.Y_PRED] if torchbearer.Y_PRED in outputs else outputs

    def on_forward_validation(self, state):
        self.on_forward(state)
