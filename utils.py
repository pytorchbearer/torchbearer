from torch.utils.data import DataLoader, TensorDataset,  Dataset
import torch
from random import shuffle as shuffle_list
from sklearn.model_selection import ShuffleSplit, KFold, LeavePOut


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
   

class CrossValidationIterator:
    def __init__(self, dataset, splitter, num_folds=1):
        super().__init__()
        self.dataset = dataset
        self.num_folds = num_folds
        self.ids = list(range(len(dataset)))
        self.current_fold = 0
        self.splitter = splitter
        self.splitter.get_n_splits(self.ids)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_fold >= self.num_folds:
            raise StopIteration
        
        train_ids, test_ids = next(self.splitter.split(self.ids))
        trainset = SubsetDataset(self.dataset, train_ids)
        valset = SubsetDataset(self.dataset, test_ids)
        self.current_fold += 1
        
        return trainset, valset


class ShuffleSplitCVIter(CrossValidationIterator):
    def __init__(self, dataset, num_folds, valid_split):
        super().__init__(dataset, ShuffleSplit(num_folds, valid_split), num_folds)
        

class KFoldCVIter(CrossValidationIterator):
    def __init__(self, dataset, num_folds):
        super().__init__(dataset, KFold(num_folds, shuffle=True), num_folds)


class LeavePOutCVIter(CrossValidationIterator):
    def __init__(self, dataset, num_folds, p):
        super().__init__(dataset, LeavePOut(p), num_folds)


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

