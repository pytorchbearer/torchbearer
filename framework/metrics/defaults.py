import framework.metrics.primitives as metrics
import framework.metrics.wrappers as wrap
import framework.metrics.running as running


def loss():
    return wrap.MetricList([running.stats(metrics.loss), wrap.stats(metrics.loss)])


def accuracy():
    return wrap.MetricList([running.stats(metrics.accuracy), wrap.stats(metrics.accuracy)])


acc = accuracy
