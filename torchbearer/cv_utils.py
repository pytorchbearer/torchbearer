import math

import torch
from torch.utils.data import TensorDataset, Dataset
import random


def train_valid_splitter(x, y, split, shuffle=True):
    """ Generate training and validation tensors from whole dataset data and label tensors
    
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
    """
    num_samples_x = x.size()[0]
    num_valid_samples = math.floor(num_samples_x * split)

    if shuffle:
        indicies = torch.randperm(num_samples_x)
        x, y = x[indicies], y[indicies]

    x_val, y_val = x[:num_valid_samples], y[:num_valid_samples]
    x, y = x[num_valid_samples:], y[num_valid_samples:]

    return x, y, x_val, y_val


def get_train_valid_sets(x, y, validation_data, validation_split, shuffle=True):
    """ Generate validation and training datasets from whole dataset tensors
    
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
    """

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


class DatasetValidationSplitter:
    def __init__(self, dataset_len, split_fraction, shuffle_seed=None):
        """ Generates training and validation split indicies for a given dataset length and creates training and
        validation datasets using these splits

        :param dataset_len: The length of the dataset to be split into training and validation
        :param split_fraction: The fraction of the whole dataset to be used for validation
        :param shuffle_seed: Optional random seed for the shuffling process
        """
        super().__init__()
        self.dataset_len = dataset_len
        self.split_fraction = split_fraction
        self.valid_ids = None
        self.train_ids = None
        self._gen_split_ids(shuffle_seed)

    def _gen_split_ids(self, seed):
        all_ids = list(range(self.dataset_len))

        if seed is not None:
            random.seed(seed)
        random.shuffle(all_ids)

        num_valid_ids = math.floor(self.dataset_len*self.split_fraction)
        self.valid_ids = all_ids[:num_valid_ids]
        self.train_ids = all_ids[num_valid_ids:]

    def get_train_dataset(self, dataset):
        """ Creates a training dataset from existing dataset

        :param dataset: Dataset to be split into a training dataset
        :type dataset: torch.utils.data.Dataset
        :return: Training dataset split from whole dataset
        :rtype: torch.utils.data.Dataset
        """
        return SubsetDataset(dataset, self.train_ids)

    def get_val_dataset(self, dataset):
        """ Creates a validation dataset from existing dataset

        :param dataset: Dataset to be split into a validation dataset
        :type dataset: torch.utils.data.Dataset
        :return: Validation dataset split from whole dataset
        :rtype: torch.utils.data.Dataset
        """
        return SubsetDataset(dataset, self.valid_ids)


class SubsetDataset(Dataset):
    def __init__(self, dataset, ids):
        """ Dataset that consists of a subset of a previous dataset

        :param dataset: Complete dataset
        :type dataset: torch.utils.data.Dataset
        :param ids: List of subset IDs
        :type ids: list
        """
        super().__init__()
        self.dataset = dataset
        self.ids = ids

    def __getitem__(self, index):
        return self.dataset.__getitem__(self.ids[index])

    def __len__(self):
        return len(self.ids)

