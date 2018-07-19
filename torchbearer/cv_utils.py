import math

import torch
from torch.utils.data import TensorDataset


def train_valid_splitter(x, y, split, shuffle=True):
    ''' Generate training and validation tensors from whole dataset data and label tensors
    
    :param x: Data tensor for whole dataset
    :type x: torch.Tensor
    :param y: Label tensor for whole dataset
    :type y: torch.Tensor
    :param split: Fraction of dataset to be used for validation
    :type split: float
    :param shuffle: If True randomize tensor order before splitting else do not randomize 
    :type shuffle: bool
    :return: Training and validation tensors (training data, training labels, validation data, validation labels)
    :rtype: tuple
    '''
    num_samples_x = x.size()[0]
    num_valid_samples = math.floor(num_samples_x * split)

    if shuffle:
        indicies = torch.randperm(num_samples_x)
        x, y = x[indicies], y[indicies]

    x_val, y_val = x[:num_valid_samples], y[:num_valid_samples]
    x, y = x[num_valid_samples:], y[num_valid_samples:]

    return x, y, x_val, y_val


def get_train_valid_sets(x, y, validation_data, validation_split, shuffle=True):
    ''' Generate validation and training datasets from whole dataset tensors
    
    :param x: Data tensor for dataset
    :type x: torch.Tensor
    :param y: Label tensor for dataset
    :type y: torch.Tensor
    :param validation_data: Optional validation data (x_val, y_val) to be used instead of splitting x and y tensors 
    :type validation_data: (torch.Tensor, torch.Tensor)
    :param validation_split: Fraction of dataset to be used for validation
    :type validation_split: float
    :param shuffle: If True randomize tensor order before splitting else do not randomize 
    :type shuffle: bool
    :return: Training and validation datasets
    :rtype: tuple
    '''

    valset = None

    if validation_data is not None:
        x_val, y_val = validation_data
    elif isinstance(validation_split, float):
        x, y, x_val, y_val = train_valid_splitter(x, y, validation_split, shuffle=shuffle)
    else:
        x_val, y_val = None, None

    trainset = TensorDataset(x, y)
    if x_val is not None and y_val is not None:
        valset = TensorDataset(x_val, y_val)

    return trainset, valset
