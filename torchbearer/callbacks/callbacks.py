class Callback(object):
    """Base callback class.

    .. note::

        All callbacks should override this class.

    """

    def on_start(self, state):
        """Perform some action with the given state as context at the start of a model fit.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_start_epoch(self, state):
        """Perform some action with the given state as context at the start of each epoch.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_start_training(self, state):
        """Perform some action with the given state as context at the start of the training loop.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_sample(self, state):
        """Perform some action with the given state as context after data has been sampled from the generator.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_forward(self, state):
        """Perform some action with the given state as context after the forward pass (model output) has been completed.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_criterion(self, state):
        """Perform some action with the given state as context after the criterion has been evaluated.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_backward(self, state):
        """Perform some action with the given state as context after backward has been called on the loss.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_step_training(self, state):
        """Perform some action with the given state as context after step has been called on the optimiser.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_end_training(self, state):
        """Perform some action with the given state as context after the training loop has completed.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_end_epoch(self, state):
        """Perform some action with the given state as context at the end of each epoch.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_end(self, state):
        """Perform some action with the given state as context at the end of the model fitting.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_start_validation(self, state):
        """Perform some action with the given state as context at the start of the validation loop.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_sample_validation(self, state):
        """Perform some action with the given state as context after data has been sampled from the validation generator.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_forward_validation(self, state):
        """Perform some action with the given state as context after the forward pass (model output) has been completed
        with the validation data.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_criterion_validation(self, state):
        """Perform some action with the given state as context after the criterion evaluation has been completed
        with the validation data.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_end_validation(self, state):
        """Perform some action with the given state as context at the end of the validation loop.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass

    def on_step_validation(self, state):
        """Perform some action with the given state as context at the end of each validation step.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        pass


class CallbackList(Callback):
    """The :class:`CallbackList` class is a wrapper for a list of callbacks which acts as a single callback.
    """

    def __init__(self, callback_list):
        """Create a new callback which wraps and internally calls each callback in the given list in turn.

        :param callback_list:The list of callbacks to be wrapped. If the list contains a :class:`CallbackList`, this
        will be unwrapped.
        :type callback_list:list

        """
        super().__init__()
        self.callback_list = []

        for i in range(len(callback_list)):
            callback = callback_list[i]

            if type(callback) is CallbackList:
                self.callback_list = self.callback_list + callback.callback_list
            else:
                self.callback_list.append(callback)

    def _for_list(self, function):
        for callback in self.callback_list:
            function(callback)
        
    def on_start(self, state):
        """Call on_start on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_start(state))

    def on_start_epoch(self, state):
        """Call on_start_epoch on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_start_epoch(state))

    def on_start_training(self, state):
        """Call on_start_training on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_start_training(state))

    def on_sample(self, state):
        """Call on_sample on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_sample(state))

    def on_forward(self, state):
        """Call on_forward on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_forward(state))

    def on_criterion(self, state):
        """Call on_criterion on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_criterion(state))

    def on_backward(self, state):
        """Call on_backward on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_backward(state))

    def on_step_training(self, state):
        """Call on_step_training on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_step_training(state))

    def on_end_training(self, state):
        """Call on_end_training on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_end_training(state))

    def on_end_epoch(self, state):
        """Call on_end_epoch on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_end_epoch(state))

    def on_end(self, state):
        """Call on_end on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_end(state))

    def on_start_validation(self, state):
        """Call on_start_validation on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_start_validation(state))

    def on_sample_validation(self, state):
        """Call on_sample_validation on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_sample_validation(state))

    def on_forward_validation(self, state):
        """Call on_forward_validation on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_forward_validation(state))

    def on_criterion_validation(self, state):
        """Call on_criterion_validation on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_criterion_validation(state))

    def on_end_validation(self, state):
        """Call on_end_validation on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_end_validation(state))

    def on_step_validation(self, state):
        """Call on_step_validation on each callback in turn with the given state.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict[str,any]

        """
        self._for_list(lambda callback: callback.on_step_validation(state))
