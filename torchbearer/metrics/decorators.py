import inspect

from torchbearer import metrics

from torchbearer.metrics import MetricFactory, EpochLambda, BatchLambda, ToDict, Mean, MetricTree, Std, RunningMean


def default_for_key(key):
    def decorator(arg):
        if inspect.isclass(arg):
            metric = arg()
        else:
            metric = arg
        metrics.DEFAULT_METRICS[key] = metric
        return arg
    return decorator


def lambda_metric(name, on_epoch=False):
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
