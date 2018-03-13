class Callback(object):

    def on_start(self, state):
        pass

    def on_start_epoch(self, state):
        pass

    def on_sample(self, state):
        pass

    def on_forward(self, state):
        pass

    def on_forward_criterion(self, state):
        pass

    def on_backward_criterion(self, state):
        pass

    def on_backward(self, state):
        pass

    def on_update_parameters(self, state):
        pass

    def on_end_epoch(self, state):
        pass

    def on_end(self, state):
        pass


class KerasCallbackWrapper(Callback):
    def __init__(self, keras_callback):
        super().__init__()
        self._keras_callback = keras_callback

    def on_start_epoch(self, state):
        return self._keras_callback.on_epoch_begin(state['epoch'], logs=state)

    def on_end_epoch(self, state):
        return self._keras_callback.on_epoch_end(state['epoch'], logs=state)

    def on_sample(self, state):
        return self._keras_callback.on_batch_begin(state['batch'], logs=state)

    def on_update_parameters(self, state):
        return self._keras_callback.on_batch_end(state['batch'], logs=state)

    def on_start(self, state):
        return self._keras_callback.on_train_begin(logs=state)

    def on_end(self, state):
        return self._keras_callback.on_end(logs=state)

