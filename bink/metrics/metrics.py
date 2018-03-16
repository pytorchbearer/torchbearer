class Metric(object):
    def __init__(self):
        super().__init__()

    def train_dict(self, state):
        return {}

    def validate_dict(self, state):
        return {}

    def final_train_dict(self, state):
        return {}

    def final_validate_dict(self, state):
        return {}

    def reset(self, state):
        pass


class MetricList(Metric):
    def __init__(self, list):
        super().__init__()
        import bink.metrics.defaults as defaults

        for i in range(len(list)):
            metric = list[i]
            if str(metric) == metric:
                list[i] = getattr(defaults, metric)()

        self._list = list

    def _for_list(self, function):
        result = {}
        for metric in self._list:
            out = function(metric)
            if out is not None:
                result.update(out)
        return result

    def train_dict(self, state):
        return self._for_list(lambda metric: metric.train_dict(state))

    def validate_dict(self, state):
        return self._for_list(lambda metric: metric.validate_dict(state))

    def final_train_dict(self, state):
        return self._for_list(lambda metric: metric.final_train_dict(state))

    def final_validate_dict(self, state):
        return self._for_list(lambda metric: metric.final_validate_dict(state))

    def reset(self, state):
        self._for_list(lambda metric: metric.reset(state))


class BasicMetric(Metric):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def train(self, state):
        pass

    def validate(self, state):
        pass

    def final_train(self, state):
        pass

    def final_validate(self, state):
        pass

    def train_dict(self, state):
        result = self.train(state)
        return {'train_' + self.name: result} if result is not None else {}

    def validate_dict(self, state):
        result = self.validate(state)
        return {'val_' + self.name: result} if result is not None else {}

    def final_train_dict(self, state):
        result = self.final_train(state)
        return {'train_' + self.name: result} if result is not None else {}

    def final_validate_dict(self, state):
        result = self.final_validate(state)
        return {'val_' + self.name: result} if result is not None else {}

    def reset(self, state):
        pass
