"""
The decorator API is the core way to interact with metrics in torchbearer. All of the classes and functionality handled
here can be reproduced by manually interacting with the classes if necessary. Broadly speaking, the decorator API is
used to construct a :class:`.MetricFactory` which will build a :class:`.MetricTree` that handles data flow between
instances of :class:`.Mean`, :class:`.RunningMean`, :class:`.Std` etc.
"""
import inspect

from torchbearer.metrics import EpochLambda, BatchLambda, ToDict, Mean, MetricTree, Std, Var, RunningMean
from .metrics import add_default


def default_for_key(key, *args, **kwargs):
    """The :func:`default_for_key` decorator will register the given metric in the global metric dict
    (`metrics.DEFAULT_METRICS`) so that it can be referenced by name in instances of :class:`.MetricList` such as in the
    list given to the :class:`.torchbearer.Model`.

    Example: ::

        @default_for_key('acc')
        class CategoricalAccuracy(metrics.BatchLambda):
            ...

    Args:
        key (str): The key to use when referencing the metric
        args: Any args to pass to the underlying metric when constructed
        kwargs: Any keyword args to pass to the underlying metric when constructed
    """
    def decorator(arg):
        add_default(key, arg, *args, **kwargs)
        return arg
    return decorator


def lambda_metric(name, on_epoch=False):
    """The :func:`lambda_metric` decorator is used to convert a lambda function `y_pred, y_true` into a :class:`.Metric`
    instance. This can be used as in the following example: ::

        @metrics.lambda_metric('my_metric')
        def my_metric(y_pred, y_true):
            ... # Calculate some metric

        model = Model(metrics=[my_metric])

    Args:
        name (str): The name of the metric (e.g. 'loss')
        on_epoch (bool): If True the metric will be an instance of :class:`.EpochLambda` instead of :class:`.BatchLambda`

    Returns:
        A decorator which replaces a function with a :class:`.Metric`
    """
    def decorator(metric_function):
        if on_epoch:
            return EpochLambda(name, metric_function)
        else:
            return BatchLambda(name, metric_function)
    return decorator


def to_dict(clazz):
    """The :func:`to_dict` decorator is used to wrap either a :class:`.Metric` class or a :class:`.Metric` instance with
    a :class:`.ToDict` instance. The result is that future output will be wrapped in a `dict[name, value]`.

    Example: ::

        >>> from torchbearer import metrics

        >>> @metrics.lambda_metric('my_metric')
        ... def my_metric(y_pred, y_true):
        ...     return y_pred + y_true
        ...
        >>> my_metric.process({'y_pred':4, 'y_true':5})
        9

        >>> @metrics.to_dict
        ... @metrics.lambda_metric('my_metric')
        ... def my_metric(y_pred, y_true):
        ...     return y_pred + y_true
        ...
        >>> my_metric.process({'y_pred':4, 'y_true':5})
        {'my_metric': 9}

    Args:
        clazz: The class to *decorate*

    Returns:
        A :class:`.ToDict` instance or a :class:`.ToDict` wrapper of the given class
    """
    if inspect.isclass(clazz):
        class Wrapper(ToDict):
            __doc__ = clazz.__doc__

            def __init__(self, *args, **kwargs):
                super(Wrapper, self).__init__(clazz(*args, **kwargs))
        Wrapper.__name__ = clazz.__name__
        return Wrapper
    else:
        return ToDict(clazz)


def _wrap_and_add_to_tree(clazz, child_func):
    if inspect.isclass(clazz):
        class Wrapper(MetricTree):
            __doc__ = clazz.__doc__

            def __init__(self, *args, **kwargs):
                inner = clazz(*args, **kwargs)
                if isinstance(inner, MetricTree):
                    super(Wrapper, self).__init__(inner.root)
                    self.children = inner.children
                else:
                    super(Wrapper, self).__init__(inner)

                self.add_child(child_func(self.root))
        Wrapper.__name__ = clazz.__name__
        return Wrapper
    else:
        inner = clazz
        if not isinstance(inner, MetricTree):
            inner = MetricTree(inner)
        inner.add_child(child_func(inner))
        return inner


def mean(clazz=None, dim=None):
    """The :func:`mean` decorator is used to add a :class:`.Mean` to the :class:`.MetricTree` which will will output a
    mean value at the end of each epoch. At build time, if the inner class is not a :class:`.MetricTree`, one will be
    created. The :class:`.Mean` will also be wrapped in a :class:`.ToDict` for simplicity.

    Example: ::

        >>> import torch
        >>> from torchbearer import metrics

        >>> @metrics.mean
        ... @metrics.lambda_metric('my_metric')
        ... def metric(y_pred, y_true):
        ...     return y_pred + y_true
        ...
        >>> metric.reset({})
        >>> metric.process({'y_pred':torch.Tensor([2]), 'y_true':torch.Tensor([2])}) # 4
        {}
        >>> metric.process({'y_pred':torch.Tensor([3]), 'y_true':torch.Tensor([3])}) # 6
        {}
        >>> metric.process({'y_pred':torch.Tensor([4]), 'y_true':torch.Tensor([4])}) # 8
        {}
        >>> metric.process_final()
        {'my_metric': 6.0}

    Args:
        clazz: The class to *decorate*
        dim (int, tuple): See :class:`.Mean`

    Returns:
        A :class:`.MetricTree` with a :class:`.Mean` appended or a wrapper class that extends :class:`.MetricTree`
    """
    if clazz is None:
        def decorator(clazz):
            return mean(clazz, dim=dim)
        return decorator

    return _wrap_and_add_to_tree(clazz, lambda metric: ToDict(Mean(metric.name, dim=dim)))


def std(clazz=None, unbiased=True, dim=None):
    """The :func:`std` decorator is used to add a :class:`.Std` to the :class:`.MetricTree` which will will output a
    sample standard deviation value at the end of each epoch. At build time, if the inner class is not a
    :class:`.MetricTree`, one will be created. The :class:`.Std` will also be wrapped in a :class:`.ToDict` (with '_std'
    appended) for simplicity.

    Example: ::

        >>> import torch
        >>> from torchbearer import metrics

        >>> @metrics.std
        ... @metrics.lambda_metric('my_metric')
        ... def metric(y_pred, y_true):
        ...     return y_pred + y_true
        ...
        >>> metric.reset({})
        >>> metric.process({'y_pred':torch.Tensor([2]), 'y_true':torch.Tensor([2])}) # 4
        {}
        >>> metric.process({'y_pred':torch.Tensor([3]), 'y_true':torch.Tensor([3])}) # 6
        {}
        >>> metric.process({'y_pred':torch.Tensor([4]), 'y_true':torch.Tensor([4])}) # 8
        {}
        >>> '%.4f' % metric.process_final()['my_metric_std']
        '2.0000'

    Args:
        clazz: The class to *decorate*
        unbiased (bool): See :class:`.Std`
        dim (int, tuple): See :class:`.Std`

    Returns:
        A :class:`.MetricTree` with a :class:`.Std` appended or a wrapper class that extends :class:`.MetricTree`
    """
    if clazz is None:
        def decorator(clazz):
            return std(clazz, unbiased=unbiased, dim=dim)
        return decorator

    return _wrap_and_add_to_tree(clazz, lambda metric: ToDict(Std(metric.name + '_std', unbiased=unbiased, dim=dim)))


def var(clazz=None, unbiased=True, dim=None):
    """The :func:`var` decorator is used to add a :class:`.Var` to the :class:`.MetricTree` which will will output a
    sample variance value at the end of each epoch. At build time, if the inner class is not a :class:`.MetricTree`, one
    will be created. The :class:`.Var` will also be wrapped in a :class:`.ToDict` (with '_var' appended) for simplicity.

    Example: ::

        >>> import torch
        >>> from torchbearer import metrics

        >>> @metrics.var
        ... @metrics.lambda_metric('my_metric')
        ... def metric(y_pred, y_true):
        ...     return y_pred + y_true
        ...
        >>> metric.reset({})
        >>> metric.process({'y_pred':torch.Tensor([2]), 'y_true':torch.Tensor([2])}) # 4
        {}
        >>> metric.process({'y_pred':torch.Tensor([3]), 'y_true':torch.Tensor([3])}) # 6
        {}
        >>> metric.process({'y_pred':torch.Tensor([4]), 'y_true':torch.Tensor([4])}) # 8
        {}
        >>> '%.4f' % metric.process_final()['my_metric_var']
        '4.0000'

    Args:
        clazz: The class to *decorate*
        unbiased (bool): See :class:`.Var`
        dim (int, tuple): See :class:`.Var`

    Returns:
        A :class:`.MetricTree` with a :class:`.Var` appended or a wrapper class that extends :class:`.MetricTree`
    """
    if clazz is None:
        def decorator(clazz):
            return var(clazz, unbiased=unbiased, dim=dim)
        return decorator

    return _wrap_and_add_to_tree(clazz, lambda metric: ToDict(Var(metric.name + '_var', unbiased=unbiased, dim=dim)))


def running_mean(clazz=None, batch_size=50, step_size=10, dim=None):
    """The :func:`running_mean` decorator is used to add a :class:`.RunningMean` to the :class:`.MetricTree`. If the
    inner class is not a :class:`.MetricTree` then one will be created. The :class:`.RunningMean` will be wrapped in a
    :class:`.ToDict` (with 'running\_' prepended to the name) for simplicity.

    .. note::
        The decorator function does not need to be called if not desired, both: `@running_mean` and `@running_mean()`
        are acceptable.

    Example: ::

        >>> import torch
        >>> from torchbearer import metrics

        >>> @metrics.running_mean(step_size=2) # Update every 2 steps
        ... @metrics.lambda_metric('my_metric')
        ... def metric(y_pred, y_true):
        ...     return y_pred + y_true
        ...
        >>> metric.reset({})
        >>> metric.process({'y_pred':torch.Tensor([2]), 'y_true':torch.Tensor([2])}) # 4
        {'running_my_metric': 4.0}
        >>> metric.process({'y_pred':torch.Tensor([3]), 'y_true':torch.Tensor([3])}) # 6
        {'running_my_metric': 4.0}
        >>> metric.process({'y_pred':torch.Tensor([4]), 'y_true':torch.Tensor([4])}) # 8, triggers update
        {'running_my_metric': 6.0}

    Args:
        clazz: The class to *decorate*
        batch_size (int): See :class:`.RunningMean`
        step_size (int): See :class:`.RunningMean`
        dim (int, tuple): See :class:`.RunningMean`

    Returns:
        decorator or :class:`.MetricTree` instance or wrapper
    """
    if clazz is None:
        def decorator(clazz):
            return running_mean(clazz, batch_size=batch_size, step_size=step_size, dim=dim)
        return decorator

    return _wrap_and_add_to_tree(clazz, lambda metric: ToDict(RunningMean('running_' + metric.name, batch_size=batch_size, step_size=step_size, dim=dim)))
