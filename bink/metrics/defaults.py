from bink import metrics


def loss():
    return metrics.MetricList([metrics.running_stats(metrics.loss_primitive), metrics.stats(metrics.loss_primitive)])


def accuracy():
    return metrics.MetricList([metrics.running_stats(metrics.categorical_accuracy_primitive), metrics.stats(metrics.categorical_accuracy_primitive)])


acc = accuracy
