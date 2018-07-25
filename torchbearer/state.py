STATE_KEYS = []


def state_key(key):
    if key in STATE_KEYS:
        count = 1
        my_key = key + '_' + str(count)

        while my_key in STATE_KEYS:
            count += 1
            my_key = key + '_' + str(count)

        key = my_key

    STATE_KEYS.append(key)
    return key


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
