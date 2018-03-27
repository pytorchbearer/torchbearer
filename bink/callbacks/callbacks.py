class Callback(object):
    def on_start(self, state):
        pass

    def on_start_epoch(self, state):
        pass

    def on_start_training(self, state):
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

    def on_step_training(self, state):
        pass

    def on_end_training(self, state):
        pass

    def on_end_epoch(self, state):
        pass

    def on_end(self, state):
        pass

    def on_start_validation(self, state):
        pass

    def on_end_validation(self, state):
        pass

    def on_step_validation(self, state):
        pass


class CallbackList(Callback):
    def __init__(self, callback_list):
        super().__init__()
        if callback_list is None:
            self._list = []
        else:
            self._list = callback_list
        
    def _for_list(self, function):
        for callback in self._list:
            function(callback)
        
    def on_start(self, state):
        self._for_list(lambda callback: callback.on_start(state))

    def on_start_epoch(self, state):
        self._for_list(lambda callback: callback.on_start_epoch(state))

    def on_start_training(self, state):
        self._for_list(lambda callback: callback.on_start_training(state))

    def on_sample(self, state):
        self._for_list(lambda callback: callback.on_sample(state))

    def on_forward(self, state):
        self._for_list(lambda callback: callback.on_forward(state))

    def on_forward_criterion(self, state):
        self._for_list(lambda callback: callback.on_forward_criterion(state))

    def on_backward_criterion(self, state):
        self._for_list(lambda callback: callback.on_backward_criterion(state))

    def on_backward(self, state):
        self._for_list(lambda callback: callback.on_backward(state))

    def on_step_training(self, state):
        self._for_list(lambda callback: callback.on_step_training(state))

    def on_end_training(self, state):
        self._for_list(lambda callback: callback.on_end_training(state))

    def on_end_epoch(self, state):
        self._for_list(lambda callback: callback.on_end_epoch(state))

    def on_end(self, state):
        self._for_list(lambda callback: callback.on_end(state))
