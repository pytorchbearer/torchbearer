torchbearer.metrics
====================================
The base metric classes exist to enable complex data flow requirements between metrics. All metrics are either instances
of :class:`.Metric` or :class:`MetricFactory`. These can then be collected in a :class:`MetricList` or a
:class:`MetricTree`. The :class:`MetricList` simply aggregates calls from a list of metrics, whereas the
:class:`MetricTree` will pass data from its root metric to each child and collect the outputs. This enables complex
running metrics and statistics, without needing to compute the underlying values more than once. Typically,
constructions of this kind should be handled using the :mod:`decorator API <.metrics.decorators>`.

Base Classes
------------------------------------

..  autoclass:: torchbearer.bases.Metric
        :members:
        :undoc-members:

..  automodule:: torchbearer.metrics.metrics
        :members:
        :undoc-members:

Decorators - The Decorator API
------------------------------------

..  automodule:: torchbearer.metrics.decorators
        :members:
        :undoc-members:

Metric Wrappers
------------------------------------

..  automodule:: torchbearer.metrics.wrappers
        :members:
        :undoc-members:

Metric Aggregators
------------------------------------

..  automodule:: torchbearer.metrics.aggregators
        :members:
        :undoc-members:

Base Metrics
------------------------------------

..  automodule:: torchbearer.metrics.default
        :members:

..  automodule:: torchbearer.metrics.primitives
        :members:

..  automodule:: torchbearer.metrics.roc_auc_score
        :members:


Timer
------------------------------------
..  automodule:: torchbearer.metrics.timer
        :members:
        :undoc-members:


