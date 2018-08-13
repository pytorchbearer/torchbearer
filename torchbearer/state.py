import warnings

__keys__ = []


def state_key(key):
    """Computes and returns a non-conflicting key for the state dictionary when given a seed key

    :param key: The seed key - basis for new state key
    :type key: String
    :return: New state key
    :rtype: String
    """
    return Statekey(key)


class Statekey:
    def __init__(self, key):
        super().__init__()
        self.key = self._gen_key_(key)

    def _gen_key_(self, key):
        if key in __keys__:
            count = 1
            my_key = key + '_' + str(count)

            while my_key in __keys__:
                count += 1
                my_key = key + '_' + str(count)

            key = my_key

        __keys__.append(key)
        return key

    def __repr__(self):
        return self.key

    def __str__(self):
        return self.key


class State(dict):
    def __init__(self):
        """
        State dictionary that behaves like a python dict but accepts Statekeys
        """
        super().__init__()

    def get_key(self, statekey):
        if isinstance(statekey, str):
            warnings.warn("State was accessed with a string, generate keys with Statekey(str).")
        return str(statekey)

    def __getitem__(self, key):
        return super().__getitem__(self.get_key(key))

    def __setitem__(self, key, val):
        super().__setitem__(self.get_key(key), val)

    def __delitem__(self, val):
        super().__delitem__(val)

    def __contains__(self, o: object) -> bool:
        return super().__contains__(self.get_key(o))

    def update(self, d):
        new_dict = {}
        for key in d:
            new_dict[self.get_key(key)] = d[key]
        super().update(new_dict)


MODEL = state_key('model')
CRITERION = state_key('criterion')
OPTIMIZER = state_key('optimizer')
DEVICE = state_key('device')
DATA_TYPE = state_key('dtype')
METRIC_LIST = state_key('metric_list')
METRICS = state_key('metrics')
SELF = state_key('self')
EPOCH = state_key('epoch')
MAX_EPOCHS = state_key('max_epochs')
TRAIN_STEPS = state_key('train_steps')
GENERATOR = state_key('generator')
TRAIN_ITERATOR = state_key('train_iterator')
VALIDATION_ITERATOR = state_key('validation_iterator')
VALIDATION_GENERATOR = state_key('validation_generator')
VALIDATION_STEPS = state_key('validation_steps')
STOP_TRAINING = state_key('stop_training')
Y_TRUE = state_key('y_true')
Y_PRED = state_key('y_pred')
X = state_key('x')
LOSS = state_key('loss')
FINAL_PREDICTIONS = state_key('final_predictions')
BATCH = state_key('t')
TIMINGS = state_key('timings')
CALLBACK_LIST = state_key('callback_list')


# d = {CALLBACK_LIST: 'test_update', MODEL: '123'}
# s = State()
# s[MODEL] = 'test'
# m1=state_key('model')
# s[m1] = 'two'
# s[state_key('model')] = 'three'
# s['ttt'] = 't'
# s.update(d)
# print(s)
# print(len(s))
