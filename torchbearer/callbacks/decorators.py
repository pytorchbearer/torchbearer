from torchbearer.callbacks import Callback


def on_start(func):
    callback = Callback()
    callback.on_start = func
    return callback


def on_start_epoch(func):
    callback = Callback()
    callback.on_start_epoch = func
    return callback


def on_start_training(func):
    new_callback = Callback()
    new_callback.on_start_training = func
    return new_callback


def on_sample(func):
    callback = Callback()
    callback.on_sample = func
    return callback


def on_forward(func):
    new_callback = Callback()
    new_callback.on_forward = func
    return new_callback


def on_criterion(func):
    new_callback = Callback()
    new_callback.on_criterion = func
    return new_callback


def on_backward(func):
    callback = Callback()
    callback.on_backward = func
    return callback


def on_step_training(func):
    callback = Callback()
    callback.on_step_training = func
    return callback


def on_end_training(func):
    callback = Callback()
    callback.on_end_training = func
    return callback


def on_end_epoch(func):
    callback = Callback()
    callback.on_end_epoch = func
    return callback


def on_end(func):
    callback = Callback()
    callback.on_end = func
    return callback


def on_start_validation(func):
    new_callback = Callback()
    new_callback.on_start_validation = func
    return new_callback


def on_sample_validation(func):
    new_callback = Callback()
    new_callback.on_sample_validation = func
    return new_callback


def on_forward_validation(func):
    new_callback = Callback()
    new_callback.on_forward_validation = func
    return new_callback


def on_end_validation(func):
    new_callback = Callback()
    new_callback.on_end_validation = func
    return new_callback


def on_step_validation(func):
    new_callback = Callback()
    new_callback.on_step_validation = func
    return new_callback
