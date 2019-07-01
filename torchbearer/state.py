from torchbearer import Metric
import warnings

__keys__ = []


def state_key(key):
    """Computes and returns a non-conflicting key for the state dictionary when given a seed key

    Args:
        key (str): The seed key - basis for new state key

    Returns:
        StateKey: New state key
    """
    return StateKey(key)


class StateKey(Metric):
    """ StateKey class that is a unique state key based on the input string key. State keys are also metrics which
    retrieve themselves from state.

    Args:
        key (str): Base key
    """
    def __init__(self, key):
        self.key = self._gen_key_(key)
        super(StateKey, self).__init__(self.key)

    def process(self, state):
        return {self.name: state[self]}

    def process_final(self, state):
        return {self.name: state[self]}

    def __call__(self, state):
        return state[self]

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
        super(State, self).__init__()

    def get_key(self, statekey):
        if isinstance(statekey, str):
            warnings.warn("State was accessed with a string: {}, generate keys with StateKey(str).".format(statekey), stacklevel=3)
        return statekey

    @property
    def data(self):
        new_state = State()
        for key in self.keys():
            try:
                new_state[key] = self[key].data
            except AttributeError:
                new_state[key] = self[key]
        return new_state

    def __getitem__(self, key):
        return super(State, self).__getitem__(self.get_key(key))

    def __setitem__(self, key, val):
        super(State, self).__setitem__(self.get_key(key), val)

    def __delitem__(self, val):
        super(State, self).__delitem__(val)

    def __contains__(self, o):
        return super(State, self).__contains__(self.get_key(o))

    def update(self, d):
        new_dict = {}
        for key in d:
            new_dict[self.get_key(key)] = d[key]
        super(State, self).update(new_dict)


#: The torchbearer version
VERSION = state_key('torchbearer_version')

#: The PyTorch module / model that will be trained
MODEL = state_key('model')

#: The criterion to use when model fitting
CRITERION = state_key('criterion')

#: The optimizer to use when model fitting
OPTIMIZER = state_key('optimizer')

#: The device currently in use by the :class:`.Trial` and PyTorch model
DEVICE = state_key('device')

#: The data type of tensors in use by the model, match this to avoid type issues
DATA_TYPE = state_key('dtype')

#: The list of metrics in use by the :class:`.Trial`
METRIC_LIST = state_key('metric_list')

#: The metric dict from the current batch of data
METRICS = state_key('metrics')

#: A self refrence to the Trial object for persistence etc.
SELF = state_key('self')

#: The current epoch number
EPOCH = state_key('epoch')

#: The total number of epochs to run for
MAX_EPOCHS = state_key('max_epochs')

#: The string name of the current data
DATA = state_key('data')

#: The current data generator (DataLoader)
GENERATOR = state_key('generator')

#: The current iterator
ITERATOR = state_key('iterator')

#: The current number of steps per epoch
STEPS = state_key('steps')

#: The train data generator in the Trial object
TRAIN_GENERATOR = state_key('train_generator')

#: The number of train steps to take
TRAIN_STEPS = state_key('train_steps')

#: The flag representing train data
TRAIN_DATA = state_key('train_data')

#: Flag for refreshing of training iterator when finished instead of each epoch
INF_TRAIN_LOADING = state_key('inf_train_loading')

#: The validation data generator in the Trial object
VALIDATION_GENERATOR = state_key('validation_generator')

#: The number of validation steps to take
VALIDATION_STEPS = state_key('validation_steps')

#: The flag representing validation data
VALIDATION_DATA = state_key('validation_data')

#: The test data generator in the Trial object
TEST_GENERATOR = state_key('test_generator')

#: The number of test steps to take
TEST_STEPS = state_key('test_steps')

#: The flag representing test data
TEST_DATA = state_key('test_data')

#: A flag that can be set to true to stop the current fit call
STOP_TRAINING = state_key('stop_training')

#: The current batch of ground truth data
TARGET = Y_TRUE = state_key('y_true')

#: The current batch of predictions
PREDICTION = Y_PRED = state_key('y_pred')

#: The current batch of inputs
INPUT = X = state_key('x')

#: The sampler which loads data from the generator onto the correct device
SAMPLER = state_key('sampler')

#: The batch loader which handles formatting data from each batch
LOADER = state_key('loader')

#: The current value for the loss
LOSS = state_key('loss')

#: The key which maps to the predictions over the dataset when calling predict
FINAL_PREDICTIONS = state_key('final_predictions')

#: The current batch number
BATCH = state_key('t')

#: The timings keys used by the timer callback
TIMINGS = state_key('timings')

#: The :class:`.CallbackList` object which is called by the Trial
CALLBACK_LIST = state_key('callback_list')

#: The history list of the Trial instance
HISTORY = state_key('history')

#: The optional arguments which should be passed to the backward call
BACKWARD_ARGS = state_key('backward_args')

#: The lambda coefficient of the linear combination of inputs
MIXUP_LAMBDA = state_key('mixup_lambda')

#: The permutation of input indices for input mixup
MIXUP_PERMUTATION = state_key('mixup_permutation')
