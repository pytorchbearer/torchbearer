import torch
from torch.utils.data import TensorDataset


def train_valid_splitter(x, y, split, shuffle=True):
    num_samples_x = x.shape[0]
    num_valid_samples = torch.floor(num_samples_x * split)

    if shuffle:
        indicies = torch.randperm(num_samples_x)
        x, y = x[indicies], y[indicies]

    x_val, y_val = x[:num_valid_samples], y[:num_valid_samples]
    x, y = x[num_valid_samples:], y[num_valid_samples:]

    return x, y, x_val, y_val


def get_train_valid_sets(x, y, validation_data, validation_split, shuffle=True):

    valset = None

    if validation_data is not None:
        x_val, y_val = validation_data
    elif validation_split > 0.0:
        x, y, x_val, y_val = train_valid_splitter(x, y, validation_split, shuffle=shuffle)
    else:
        x_val, y_val = None, None

    trainset = TensorDataset(x, y)
    if x_val is not None and y_val is not None:
        valset = TensorDataset(x_val, y_val)

    return trainset, valset
