from torchbearer import metrics


def loss():
    """Return the default loss metric, running / overall mean and std.

    :return: Metric -- Running and final loss metrics.

    """
    return metrics.MetricList([metrics.running_stats(metrics.loss_primitive), metrics.stats(metrics.loss_primitive)])


def accuracy():
    """Return the default accuracy metric, running / overall mean and std of categorical accuracy.

    :return: Metric -- Running and final categorical accuracy metrics.

    """
    return metrics.MetricList([metrics.running_stats(metrics.categorical_accuracy_primitive), metrics.stats(metrics.categorical_accuracy_primitive)])


acc = accuracy
