from torchbearer import Callback


class CallbackList(Callback):
    """The :class:`CallbackList` class is a wrapper for a list of callbacks which acts as a single :class:`.Callback` and
    internally calls each :class:`.Callback` in the given list in turn.

    Args:
        callback_list (list): The list of callbacks to be wrapped. If the list contains a :class:`CallbackList`, this
            will be unwrapped.
    """

    CALLBACK_STATES = 'callback_states'
    CALLBACK_TYPES = 'callback_types'

    def __init__(self, callback_list):
        super(CallbackList, self).__init__()
        self.callback_list = []
        self.append(callback_list)

    def state_dict(self):
        """Get a dict containing all of the callback states.

        Returns:
            dict: A dict containing parameters and persistent buffers.
        """
        state_dict = {
            CallbackList.CALLBACK_STATES: [],
            CallbackList.CALLBACK_TYPES: []
        }

        def to_state(callback):
            state_dict[CallbackList.CALLBACK_STATES].append(callback.state_dict())
            state_dict[CallbackList.CALLBACK_TYPES].append(callback.__class__)

        self._for_list(to_state)

        return state_dict

    def load_state_dict(self, state_dict):
        """Resume this callback list from the given state. Callbacks must be given in the same order for this to work.

        Args:
            state_dict (dict): The state dict to reload

        Returns:
            CallbackList: self
        """

        t_iter = iter(state_dict[CallbackList.CALLBACK_TYPES])
        s_iter = iter(state_dict[CallbackList.CALLBACK_STATES])

        def from_state(callback):
            if callback.__class__ == next(t_iter):
                callback.load_state_dict(next(s_iter))
            else:
                import warnings
                warnings.warn('Callback classes did not match, expected: ' + str([c.__name__ for c in state_dict[CallbackList.CALLBACK_TYPES]]))

        self._for_list(from_state)

        return self

    def _for_list(self, function):
        for callback in self.callback_list:
            function(callback)

    def __str__(self):
        return str([str(c) for c in self.callback_list])

    def __iter__(self):
        return self.callback_list.__iter__()

    def __copy__(self):
        return CallbackList(self.callback_list)

    def copy(self):
        return self.__copy__()

    def append(self, callback_list):
        for callback in callback_list:
            if isinstance(callback, CallbackList):
                self.callback_list = self.callback_list + callback.callback_list
            else:
                self.callback_list.append(callback)

    def on_init(self, state):
        """Call on_init on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_init(state))

    def on_start(self, state):
        """Call on_start on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_start(state))

    def on_start_epoch(self, state):
        """Call on_start_epoch on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_start_epoch(state))

    def on_start_training(self, state):
        """Call on_start_training on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_start_training(state))

    def on_sample(self, state):
        """Call on_sample on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_sample(state))

    def on_forward(self, state):
        """Call on_forward on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_forward(state))

    def on_criterion(self, state):
        """Call on_criterion on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_criterion(state))

    def on_backward(self, state):
        """Call on_backward on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_backward(state))

    def on_step_training(self, state):
        """Call on_step_training on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_step_training(state))

    def on_end_training(self, state):
        """Call on_end_training on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_end_training(state))

    def on_start_validation(self, state):
        """Call on_start_validation on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_start_validation(state))

    def on_sample_validation(self, state):
        """Call on_sample_validation on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_sample_validation(state))

    def on_forward_validation(self, state):
        """Call on_forward_validation on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_forward_validation(state))

    def on_criterion_validation(self, state):
        """Call on_criterion_validation on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_criterion_validation(state))

    def on_step_validation(self, state):
        """Call on_step_validation on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_step_validation(state))

    def on_end_validation(self, state):
        """Call on_end_validation on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_end_validation(state))

    def on_end_epoch(self, state):
        """Call on_end_epoch on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_end_epoch(state))

    def on_checkpoint(self, state):
        """Call on_checkpoint on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_checkpoint(state))

    def on_end(self, state):
        """Call on_end on each callback in turn with the given state.

        Args:
            state (dict[str,any]): The current state dict of the :class:`.Trial`.
        """
        self._for_list(lambda callback: callback.on_end(state))
