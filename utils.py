from torch.utils.data import DataLoader, TensorDataset,  Dataset
import torch
from random import shuffle as shuffle_list


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


class DatasetCrossValidationIter:
    def __init__(self, dataset, num_folds=1, valid_split=0.1, shuffle=False):
        super().__init__()
        self.dataset = dataset
        self.num_folds = num_folds
        self.valid_len = int(len(dataset)*valid_split)
        self.ids = list(range(len(dataset)))
        self.current_fold = 0

        if shuffle:
            shuffle_list(self.ids)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_fold >= self.num_folds:
            raise StopIteration

        val_start = self.current_fold*self.valid_len
        valid_ids = self.ids[val_start:val_start+self.valid_len]

        sets = DatasetValidationSplitter(self.dataset, valid_ids)
        self.current_fold += 1
        return sets.get_train_dataset(), sets.get_valid_dataset()


class DatasetValidationSplitter:
    def __init__(self, dataset, valid_ids, shuffle=False):
        super().__init__()
        self.valid_ids = valid_ids
        self.train_ids = [x for x in range(len(dataset)) if x not in valid_ids]
        self.dataset = dataset
        self.shuffle=shuffle

    def get_train_dataset(self):
        return SubsetDataset(self.dataset, self.train_ids, shuffle=self.shuffle)

    def get_valid_dataset(self):
        return SubsetDataset(self.dataset, self.valid_ids, shuffle=self.shuffle)


class SubsetDataset(Dataset):
    def __init__(self, dataset, ids, shuffle=False):
        super().__init__()
        self.dataset = dataset
        self.ids = ids

        if shuffle:
            shuffle_list(self.ids)

    def __getitem__(self, index):
        return self.dataset.__getitem__(self.ids[index])

    def __len__(self):
        return len(self.ids)

