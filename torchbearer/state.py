import warnings

__keys__ = []


def state_key(key):
    """Computes and returns a non-conflicting key for the state dictionary when given a seed key

    :param key: The seed key - basis for new state key
    :type key: String
    :return: New state key
    :rtype: String
    """
    return StateKey(key)


class StateKey:
    """ StateKey class that is a unique state key based of input string key

    :param key: Base key
    """
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

    def __eq__(self, other):
        return self.key == str(other)
        
    def __hash__(self):
        return self.key.__hash__()


class State(dict):
    """
    State dictionary that behaves like a python dict but accepts StateKeys
    """
    def __init__(self):
        super().__init__()

    def get_key(self, statekey):
        if isinstance(statekey, str):
            warnings.warn("State was accessed with a string: {}, generate keys with StateKey(str).".format(statekey), stacklevel=2)
        return statekey

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


VERSION = state_key('torchbearer_version')

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

GENERATOR = state_key('generator')
ITERATOR = state_key('iterator')
STEPS = state_key('steps')

TRAIN_GENERATOR = state_key('train_generator')
TRAIN_STEPS = state_key('train_steps')
TRAIN_DATA = state_key('train_data')

VALIDATION_GENERATOR = state_key('validation_generator')
VALIDATION_STEPS = state_key('validation_steps')
VALIDATION_DATA = state_key('validation_data')

TEST_GENERATOR = state_key('test_generator')
TEST_STEPS = state_key('test_steps')
TEST_DATA = state_key('test_data')

STOP_TRAINING = state_key('stop_training')
Y_TRUE = state_key('y_true')
Y_PRED = state_key('y_pred')
X = state_key('x')
SAMPLER = state_key('sampler')
LOSS = state_key('loss')
FINAL_PREDICTIONS = state_key('final_predictions')
BATCH = state_key('t')
TIMINGS = state_key('timings')
CALLBACK_LIST = state_key('callback_list')
HISTORY = state_key('history')
BACKWARD_ARGS = state_key('backward_args')

# Legacy
VALIDATION_ITERATOR = 'validation_iterator'
TRAIN_ITERATOR = 'train_iterator'
