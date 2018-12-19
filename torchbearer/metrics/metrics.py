"""
The base metric classes exist to enable complex data flow requirements between metrics. All metrics are either instances
of :class:`.Metric` or :class:`MetricFactory`. These can then be collected in a :class:`MetricList` or a
:class:`MetricTree`. The :class:`MetricList` simply aggregates calls from a list of metrics, whereas the
:class:`MetricTree` will pass data from its root metric to each child and collect the outputs. This enables complex
running metrics and statistics, without needing to compute the underlying values more than once. Typically,
constructions of this kind should be handled using the :mod:`decorator API <.metrics.decorators>`.

..  autoclass:: torchbearer.bases.Metric
        :members:
        :undoc-members:
"""
import inspect
from torchbearer import Metric

__defaults__ = {}


def add_default(key, metric, *args, **kwargs):
    __defaults__[key] = (metric, args, kwargs)


def get_default(key):
    metric, args, kwargs = __defaults__[key]
    if inspect.isclass(metric):
        metric = metric(*args, **kwargs)
    return metric


class MetricTree(Metric):
    """A tree structure which has a node :class:`.Metric` and some children. Upon execution, the node is called with the
    input and its output is passed to each of the children. A dict is updated with the results.

    .. note::

       If the node output is already a dict (i.e. the node is a standalone metric), this is unwrapped before passing the
       **first** value to the children.

    :param metric: The metric to act as the root node of the tree / subtree
    :type metric: Metric
    """
    def __init__(self, metric):
        super(MetricTree, self).__init__(metric.name)
        self.root = metric
        self.children = []

    def __str__(self):
        return str(self.root)

    def add_child(self, child):
        """Add a child to this node of the tree

        :param child: The child to add
        :type child: Metric
        :return: None
        """
        self.children.append(child)

    def _for_tree(self, function, *args):
        result = {}
        node_value = function(self.root, *args)
        if isinstance(node_value, dict):
            node_value = next(iter(node_value.values()))

        for subtree in self.children:
            out = function(subtree, node_value)
            if out is not None:
                result.update(out)

        return result

    def process(self, *args):
        """Process this node and then pass the output to each child.

        :return: A dict containing all results from the children
        """
        return self._for_tree(lambda metric, *in_args: metric.process(*in_args), *args)

    def process_final(self, *args):
        """Process this node and then pass the output to each child.

        :return: A dict containing all results from the children
        """
        return self._for_tree(lambda metric, *in_args: metric.process_final(*in_args), *args)

    def eval(self, data_key=None):
        self.root.eval(data_key=data_key)

        for subtree in self.children:
            subtree.eval(data_key=data_key)

    def train(self):
        self.root.train()

        for subtree in self.children:
            subtree.train()

    def reset(self, state):
        self.root.reset(state)

        for subtree in self.children:
            subtree.reset(state)


class MetricList(Metric):
    """The :class:`MetricList` class is a wrapper for a list of metrics which acts as a single metric and produces a
    dictionary of outputs.

    :param metric_list: The list of metrics to be wrapped. If the list contains a :class:`MetricList`, this will be\
    unwrapped. Any strings in the list will be retrieved from metrics.DEFAULT_METRICS.
    :type metric_list: list
    """

    def __init__(self, metric_list):
        super().__init__('metric_list')

        self.metric_list = []

        for metric in metric_list:

            if isinstance(metric, str):
                metric = get_default(metric)

            if isinstance(metric, MetricList):
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

    def process(self, *args):
        """Process each metric an wrap in a dictionary which maps metric names to values.

        :return: dict[str,any] -- A dictionary which maps metric names to values.

        """
        return self._for_list(lambda metric: metric.process(*args))

    def process_final(self, *args):
        """Process each metric an wrap in a dictionary which maps metric names to values.

        :return: dict[str,any] -- A dictionary which maps metric names to values.

        """
        return self._for_list(lambda metric: metric.process_final(*args))

    def train(self):
        """Put each metric in train mode.
        """
        self._for_list(lambda metric: metric.train())

    def eval(self, data_key=None):
        """Put each metric in eval mode
        """
        self._for_list(lambda metric: metric.eval(data_key=data_key))

    def reset(self, state):
        """Reset each metric with the given state.

        :param state: The current state dict of the :class:`.Model`.

        """
        self._for_list(lambda metric: metric.reset(state))

    def __str__(self):
        return str([str(m) for m in self.metric_list])


class AdvancedMetric(Metric):
    """The :class:`AdvancedMetric` class is a metric which provides different process methods for training and
    validation. This enables running metrics which do not output intermediate steps during validation.

    :param name: The name of the metric.
    :type name: str

    """

    def __init__(self, name):
        super(AdvancedMetric, self).__init__(name)
        self._process = self.process_train
        self._process_final = self.process_final_train

    def process_train(self, *args):
        """Process the given state and return the metric value for a training iteration.

        :return: The metric value for a training iteration.

        """
        pass

    def process_validate(self, *args):
        """Process the given state and return the metric value for a validation iteration.

        :return: The metric value for a validation iteration.

        """
        pass

    def process_final_train(self, *args):
        """Process the given state and return the final metric value for a training iteration.

        :return: The final metric value for a training iteration.

        """
        pass

    def process_final_validate(self, *args):
        """Process the given state and return the final metric value for a validation iteration.

        :return: The final metric value for a validation iteration.

        """
        pass

    def process(self, *args):
        """Depending on the current mode, return the result of either 'process_train' or 'process_validate'.

        :return: The metric value.

        """
        return self._process(*args)

    def process_final(self, *args):
        """Depending on the current mode, return the result of either 'process_final_train' or 'process_final_validate'.

        :return: The final metric value.

        """
        return self._process_final(*args)

    def eval(self, data_key=None):
        """Put the metric in eval mode.

        :param data_key: The torchbearer data_key, if used
        :type data_key: Optional(torchbearer.StateKey)

        """
        self._process = self.process_validate
        self._process_final = self.process_final_validate

    def train(self):
        """Put the metric in train mode.
        """
        self._process = self.process_train
        self._process_final = self.process_final_train
