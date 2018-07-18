class Metric(object):
    """Base metric class.

    .. note::

        All metrics must override this class.

    """

    def __init__(self, name):
        """Construct an empty metric with the given name.

        :param name: The name of the metric to be used in outputs and logs.
        :type name: str

        """
        super().__init__()
        self.name = name

    def process(self, state):
        """Process the state and update the metric for one iteration.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict
        :return: None or the current value of the metric.

        """
        ...

    def process_final(self, state):
        """Process the terminal state and output the final value of the metric.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict
        :return: None or the final value of the metric.

        """
        ...

    def eval(self):
        """Put the metric in eval mode during model validation.
        """
        ...

    def train(self):
        """Put the metric in train mode during model training.
        """
        ...

    def reset(self, state):
        """Reset the metric, called before the start of an epoch.

        :param state: The current state dict of the :class:`Model`.

        """
        ...


class MetricList(Metric):
    """The :class:`MetricList` class is a wrapper for a list of metrics which acts as a single metric and produces a
    dictionary of outputs.
    """

    def __init__(self, metric_list):
        """Create a new metric which wraps and internally calls each metric in the given list in turn.

        :param metric_list:The list of metrics to be wrapped. If the list contains a :class:`MetricList`, this will be\
        unwrapped. Any strings in the list will be retrieved from :mod:`defaults`.
        :type metric_list:list

        """
        super().__init__('metric_list')
        import torchbearer.metrics.defaults as defaults

        self.prefix = ''
        self.metric_list = []

        for i in range(len(metric_list)):
            metric = metric_list[i]

            if str(metric) == metric:
                metric = getattr(defaults, metric)()

            if type(metric) is MetricList:
                self.metric_list = self.metric_list + metric.metric_list
            else:
                self.metric_list.append(metric)

    def _for_list(self, function):
        result = {}
        for metric in self.metric_list:
            out = function(metric)
            if out is not None:
                result.update(out)
        return result

    def _to_dict(self, name, result):
        return {self.prefix + name: result} if result is not None else None

    def process(self, state):
        """Process each metric an wrap in a dictionary which maps metric names to values.

        :param state: The current state dict of the :class:`Model`.
        :return: dict[str,any] -- A dictionary which maps metric names to values.

        """
        return self._for_list(lambda metric: self._to_dict(metric.name, metric.process(state)))

    def process_final(self, state):
        """Process each metric an wrap in a dictionary which maps metric names to values.

        :param state: The current state dict of the :class:`Model`.
        :return: dict[str,any] -- A dictionary which maps metric names to values.

        """
        return self._for_list(lambda metric: self._to_dict(metric.name, metric.process_final(state)))

    def train(self):
        """Put each metric in train mode.
        """
        self.prefix = ''
        self._for_list(lambda metric: metric.train())

    def eval(self):
        """Put each metric in eval mode and prepend 'val\_' to future outputs.
        """
        self.prefix = 'val_'
        self._for_list(lambda metric: metric.eval())

    def reset(self, state):
        """Reset each metric with the given state.

        :param state: The current state dict of the :class:`Model`.

        """
        self._for_list(lambda metric: metric.reset(state))


class AdvancedMetric(Metric):
    """The :class:`AdvancedMetric` class is a metric which provides different process methods for training and
    validation. This enables running metrics which do not output intermediate steps during validation.
    """

    def __init__(self, name):
        """Construct a new :class:`AdvancedMetric` with the given name.

        :param name: The name of the metric.
        :type name: str

        """
        super().__init__(name)
        self._process = self.process_train
        self._process_final = self.process_final_train

    def process_train(self, state):
        """Process the given state and return the metric value for a training iteration.

        :param state: The current state dict of the :class:`Model`.
        :return: The metric value for a training iteration.

        """
        ...

    def process_validate(self, state):
        """Process the given state and return the metric value for a validation iteration.

        :param state: The current state dict of the :class:`Model`.
        :return: The metric value for a validation iteration.

        """
        ...

    def process_final_train(self, state):
        """Process the given state and return the final metric value for a training iteration.

        :param state: The current state dict of the :class:`Model`.
        :return: The final metric value for a training iteration.

        """
        ...

    def process_final_validate(self, state):
        """Process the given state and return the final metric value for a validation iteration.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict
        :return: The final metric value for a validation iteration.

        """
        ...

    def process(self, state):
        """Depending on the current mode, return the result of either 'process_train' or 'process_validate'.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict
        :return: The metric value.

        """
        return self._process(state)

    def process_final(self, state):
        """Depending on the current mode, return the result of either 'process_final_train' or 'process_final_validate'.

        :param state: The current state dict of the :class:`Model`.
        :type state: dict
        :return: The final metric value.

        """
        return self._process_final(state)

    def eval(self):
        """Put the metric in eval mode.
        """
        self._process = self.process_validate
        self._process_final = self.process_final_validate

    def train(self):
        """Put the metric in train mode.
        """
        self._process = self.process_train
        self._process_final = self.process_final_train
