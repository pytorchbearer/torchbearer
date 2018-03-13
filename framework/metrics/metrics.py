from abc import ABC, abstractmethod


class MetricBase(ABC):
    def __init__(self):
        super().__init__()
        self.reset()

    @abstractmethod
    def process_train(self, state):
        pass

    @abstractmethod
    def process_val(self, state):
        pass

    @abstractmethod
    def process_final(self, state):
        pass

    @abstractmethod
    def reset(self):
        pass


class LossMetric(MetricBase):
    def __init__(self):
        super().__init__()
        self._name = 'Loss'
        self._total = 0.0
        self._count = 0

    def process_train(self, state):
        return {self._name: self._process(state)}

    def process_val(self, state):
        return {self._name: self._process(state)}

    def _process(self, state):
        result = state['loss']
        self._total += result * state['y_pred'].shape[0]
        self._count += state['y_pred'].shape[0]

        return result

    def process_final(self, state):
        return self._total / self._count

    def reset(self):
        self._total = 0.0
        self._count = 0


class SimpleMetric(MetricBase):
    def __init__(self, name, metric_function):
        super().__init__()
        self._name = name
        self._metric_function = metric_function

    def process_train(self, state):
        return {self._name: self._process(state)}

    def process_val(self, state):
        return {self._name: self._process(state)}

    def _process(self, state):
        return self._metric_function(state['y_true'], state['y_pred'])

    def _process_final(self, state):
        pass

    def process_final(self, state):
        result = self._process_final(state)
        if result:
            return {self._name: result}

    def reset(self):
        pass


# TODO: better name for v
class SimpleAveragingMetric(SimpleMetric):
    def __init__(self, name, metric_function, v=0.0):
        super().__init__(name, metric_function)
        self._v = v
        self._total = v
        self._count = 0

    def _process(self, state):
        result = super()._process(state)
        self._total += result * state['y_pred'].shape[0]
        self._count += state['y_pred'].shape[0]

        return result

    def _process_final(self, state):
        return self._total / self._count

    def reset(self):
        self._total = self._v
        self._count = 0
