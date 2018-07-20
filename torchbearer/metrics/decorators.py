"""
The decorator API is the core way to interact with metrics in torchbearer. All of the classes and functionality handled
here can be reproduced by manually interacting with the classes if necessary. Broadly speaking, the decorator API is
used to construct a :class:`.MetricFactory` which will build a :class:`.MetricTree` that handles data flow between
instances of :class:`.Mean`, :class:`.RunningMean`, :class:`.Std` etc.
"""

import inspect

from torchbearer import metrics

from torchbearer.metrics import MetricFactory, EpochLambda, BatchLambda, ToDict, Mean, MetricTree, Std, RunningMean


def default_for_key(key):
    """The :func:`default_for_key` decorator will register the given metric in the global metric dict
    (`metrics.DEFAULT_METRICS`) so that it can be referenced by name in instances of :class:`.MetricList` such as in the
    list given to the :class:`.torchbearer.Model`.

    Example: ::

        @default_for_key('acc')
        class CategoricalAccuracy(metrics.BatchLambda):
            ...

    :param key: The key to use when referencing the metric
    :type key: str

    """
    def decorator(arg):
        if inspect.isclass(arg):
            metric = arg()
        else:
            metric = arg
        metrics.DEFAULT_METRICS[key] = metric
        return arg
    return decorator


def lambda_metric(name, on_epoch=False):
    """The :func:`lambda_metric` decorator is used to convert a lambda function `y_pred, y_true` into a :class:`.Metric`
    instance. In fact it return a :class:`.MetricFactory` which is used to build a :class:`.Metric`. This can make
    things complicated as in the following example: ::

        @metrics.lambda_metric('my_metric')
        def my_metric(y_pred, y_true):
            ... # Calculate some metric

        model = Model(metrics=[my_metric()]) # Note we have to call `my_metric` in order to instantiate the class

    :param name: The name of the metric (e.g. 'loss')
    :param on_epoch: If True the metric will be an instance of :class:`.EpochLambda` instead of :class:`.BatchLambda`
    :return: A decorator which replaces a function with a :class:`.MetricFactory`
    """
    def decorator(metric_function):
        class LambdaFactory(MetricFactory):
            def build(self):
                if on_epoch:
                    return EpochLambda(name, metric_function)
                else:
                    return BatchLambda(name, metric_function)

        return LambdaFactory
    return decorator


def to_dict(clazz):
    """The :func:`to_dict` decorator is used to wrap either a :class:`.Metric` or :class:`.MetricFactory` instance with
    a :class:`.ToDict` instance. The result is that future output will be wrapped in a `dict[name, value]`.

    Example: ::

        >>> from torchbearer import metrics

        >>> @metrics.lambda_metric('my_metric')
        ... def my_metric(y_pred, y_true):
        ...     return y_pred + y_true
        ...
        >>> my_metric().build().process({'y_pred':4, 'y_true':5})
        9

        >>> @metrics.to_dict
        ... @metrics.lambda_metric('my_metric')
        ... def my_metric(y_pred, y_true):
        ...     return y_pred + y_true
        ...
        >>> my_metric().build().process({'y_pred':4, 'y_true':5})
        {'my_metric': 9}

    :param clazz: The class to *decorate*
    :return: A :class:`.MetricFactory` which can be instantiated and built to wrap the given class in a :class:`.ToDict`
    """
    class DictFactory(MetricFactory):
        def __init__(self, *args, **kwargs):
            self.inner = clazz(*args, **kwargs)

        def build(self):
            if isinstance(self.inner, MetricFactory):
                inner = self.inner.build()
            else:
                inner = self.inner

            return ToDict(inner)

    return DictFactory


def mean(clazz):
    """The :func:`mean` decorator is used to add a :class:`.Mean` to the :class:`.MetricTree` which will will output a
    mean value at the end of each epoch. At build time, if the inner class is not a :class:`.MetricTree`, one will be
    created. The :class:`.Mean` will also be wrapped in a :class:`.ToDict` for simplicity.

    Example: ::

        >>> import torch
        >>> from torchbearer import metrics

        >>> @metrics.mean
        ... @metrics.lambda_metric('my_metric')
        ... def my_metric(y_pred, y_true):
        ...     return y_pred + y_true
        ...
        >>> metric = my_metric().build()
        >>> metric.reset({})
        >>> metric.process({'y_pred':torch.Tensor([2]), 'y_true':torch.Tensor([2])}) # 4
        {}
        >>> metric.process({'y_pred':torch.Tensor([3]), 'y_true':torch.Tensor([3])}) # 6
        {}
        >>> metric.process({'y_pred':torch.Tensor([4]), 'y_true':torch.Tensor([4])}) # 8
        {}
        >>> metric.process_final()
        {'my_metric': 6.0}

    :param clazz: The class to *decorate*
    :return: A :class:`.MetricFactory` which can be instantiated and built to append a :class:`.Mean` to the\
    :class:`.MetricTree`
    """
    class MeanFactory(MetricFactory):
        def __init__(self, *args, **kwargs):
            self.inner = clazz(*args, **kwargs)

        def build(self):
            if isinstance(self.inner, MetricFactory):
                inner = self.inner.build()
            else:
                inner = self.inner

            if not isinstance(inner, MetricTree):
                inner = MetricTree(inner)
            inner.add_child(ToDict(Mean(inner.name)))
            return inner

    return MeanFactory


def std(clazz):
    """The :func:`std` decorator is used to add a :class:`.Std` to the :class:`.MetricTree` which will will output a
    population standard deviation value at the end of each epoch. At build time, if the inner class is not a
    :class:`.MetricTree`, one will be created. The :class:`.Std` will also be wrapped in a :class:`.ToDict` (with '_std'
    appended) for simplicity.

    Example: ::

        >>> import torch
        >>> from torchbearer import metrics

        >>> @metrics.std
        ... @metrics.lambda_metric('my_metric')
        ... def my_metric(y_pred, y_true):
        ...     return y_pred + y_true
        ...
        >>> metric = my_metric().build()
        >>> metric.reset({})
        >>> metric.process({'y_pred':torch.Tensor([2]), 'y_true':torch.Tensor([2])}) # 4
        {}
        >>> metric.process({'y_pred':torch.Tensor([3]), 'y_true':torch.Tensor([3])}) # 6
        {}
        >>> metric.process({'y_pred':torch.Tensor([4]), 'y_true':torch.Tensor([4])}) # 8
        {}
        >>> '%.4f' % metric.process_final()['my_metric_std']
        '1.6330'

    :param clazz: The class to *decorate*
    :return: A :class:`.MetricFactory` which can be instantiated and built to append a :class:`.Mean` to the\
    :class:`.MetricTree`
    """
    class StdFactory(MetricFactory):
        def __init__(self, *args, **kwargs):
            self.inner = clazz(*args, **kwargs)

        def build(self):
            if isinstance(self.inner, MetricFactory):
                inner = self.inner.build()
            else:
                inner = self.inner

            if not isinstance(inner, MetricTree):
                inner = MetricTree(inner)
            inner.add_child(ToDict(Std(inner.name + '_std')))
            return inner

    return StdFactory


def running_mean(clazz=None, batch_size=50, step_size=10):
    """The :func:`running_mean` decorator is used to add a :class:`.RunningMean` to the :class:`.MetricTree`. As with the
    other decorators, a :class:`.MetricFactory` is created which will do this upon the call to
    :meth:`.MetricFactory.build`. If the inner class is not / does not build a :class:`.MetricTree` then one will be
    created. The :class:`.RunningMean` will be wrapped in a :class:`.ToDict` (with 'running\_' prepended to the name) for
    simplicity.

    .. note::
        The decorator function does not need to be called if not desired, both: `@running_mean` and `@running_mean()`
        are acceptable.

    Example: ::

        >>> import torch
        >>> from torchbearer import metrics

        >>> @metrics.running_mean(step_size=2) # Update every 2 steps
        ... @metrics.lambda_metric('my_metric')
        ... def my_metric(y_pred, y_true):
        ...     return y_pred + y_true
        ...
        >>> metric = my_metric().build()
        >>> metric.reset({})
        >>> metric.process({'y_pred':torch.Tensor([2]), 'y_true':torch.Tensor([2])}) # 4
        {'running_my_metric': 4.0}
        >>> metric.process({'y_pred':torch.Tensor([3]), 'y_true':torch.Tensor([3])}) # 6
        {'running_my_metric': 4.0}
        >>> metric.process({'y_pred':torch.Tensor([4]), 'y_true':torch.Tensor([4])}) # 8, triggers update
        {'running_my_metric': 6.0}

    :param clazz: The class to *decorate*
    :param batch_size: See :class:`.RunningMean`
    :param step_size: See :class:`.RunningMean`
    :return: decorator or :class:`.MetricFactory`
    """
    class RunningMeanFactory(MetricFactory):
        def __init__(self, *args, **kwargs):
            self.inner = clazz(*args, **kwargs)

        def build(self):
            if isinstance(self.inner, MetricFactory):
                inner = self.inner.build()
            else:
                inner = self.inner

            if not isinstance(inner, MetricTree):
                inner = MetricTree(inner)
            inner.add_child(ToDict(RunningMean('running_' + inner.name, batch_size=batch_size, step_size=step_size)))
            return inner

    if clazz is None:
        def decorator(clazz):
            return running_mean(clazz, batch_size=batch_size, step_size=step_size)
        return decorator
    return RunningMeanFactory
