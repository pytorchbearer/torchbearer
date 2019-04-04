from torch.utils.data import Dataset, DataLoader
import random
import math
import torchbearer as tb
from torchbearer.callbacks import Callback
from torchbearer.bases import fluent


class TrainValTestSplit(Callback):
    def __init__(self, dataset, train_size=None, val_size=None, test_size=None, seed=None, shuffle=True):
        super().__init__()
        self.dataset = dataset
        self.train_size, self.val_size, self.test_size = self._get_split_sizes(*self._size_to_frac(train_size, val_size, 
                                                                                                   test_size))
        self.seed = seed
        self.shuffle = shuffle
        self.dl_args = []
        self.dl_kwargs = {}

    def _size_to_frac(self, train_size, val_size, test_size):
        def size_to_float(size):
            return size / len(self.dataset) if size is not None and size > 1.0 else size
        
        train_size = size_to_float(train_size)
        val_size = size_to_float(val_size)
        test_size = size_to_float(test_size)

        return train_size, val_size, test_size

    def _get_split_sizes(self, train_size, val_size, test_size):

        if train_size is None and val_size is not None:
            val_size, train_size, test_size = self._infer_split(val_size, train_size, test_size)
        if train_size is not None:
            train_size, val_size, test_size = self._infer_split(train_size, val_size, test_size)
        
        if train_size is None and val_size is None:
            if test_size is None:
                train_size = 0.8
                val_size = 0.2
            else:
                train_size = 1 - test_size
        
        train_size = 0.0 if train_size is None else train_size
        val_size = 0.0 if val_size is None else val_size
        test_size = 0.0 if test_size is None else test_size
        return train_size, val_size, test_size

    @staticmethod
    def _infer_split(main_size, second_size, test_size):
        """Infers split fractions when given a defined split value (main_size) and a known or unknown second_size. 
        main_size and second_size are one of train_size or val_size
        
        Args:
            main_size (float): One of train_size or val_size
            second_size(None or float): Opposing size to main_size (train or val)
            test_size (None or float): Test Size 

        Returns (tuple): main_size, second_size, test_size 

        """
        if test_size is None:
            if second_size is None:
                second_size = 1.0 - main_size
                return main_size, second_size, test_size
            elif main_size + second_size < 1.0:
                test_size = 1 - (main_size + second_size)
                return main_size, second_size, test_size
            return main_size, second_size, test_size

        if second_size is None:
            if main_size + test_size < 1.0:
                second_size = 1 - (main_size + test_size)
                return main_size, second_size, test_size

        return main_size, second_size, test_size

    @fluent
    def with_dataloader_parameters(self, *args, **kwargs):
        self.dl_args = args
        self.dl_kwargs = kwargs
            
    def on_init(self, state):
        super().on_init(state)
        splitter = DatasetValidationSplitter(len(self.dataset), self.val_size, self.test_size, self.shuffle, self.seed)
        trainset, valset, testset = splitter.get_datasets(self.dataset)
        trainloader = DataLoader(trainset, *self.dl_args, **self.dl_kwargs)
        valloader = DataLoader(valset, *self.dl_args, **self.dl_kwargs)
        testloader = DataLoader(testset, *self.dl_args, **self.dl_kwargs)

        state[tb.SELF].with_generators(trainloader, valloader, testloader)

    def state_dict(self):
        return super().state_dict()

    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict)


class DatasetValidationSplitter:
    def __init__(self, dataset_len, val_split, test_split, shuffle=True, shuffle_seed=None):
        """ Generates training and validation split indicies for a given dataset length and creates training and
        validation datasets using these splits

        Args:
            dataset_len: The length of the dataset to be split into training and validation
            split_fraction: The fraction of the whole dataset to be used for validation
            shuffle_seed: Optional random seed for the shuffling process
        """
        self.dataset_len = dataset_len
        self.val_split = val_split
        self.test_split = test_split
        self.test_ids = None
        self.valid_ids = None
        self.train_ids = None
        self._gen_split_ids(shuffle_seed)
        self.shuffle = shuffle

    def _gen_split_ids(self, seed):
        all_ids = list(range(self.dataset_len))

        if seed is not None:
            random.seed(seed)

        if self.shuffle:
            random.shuffle(all_ids)

        num_valid_ids = int(math.floor(self.dataset_len*self.val_split))
        num_test_ids = int(math.floor(self.dataset_len*self.test_split))

        self.valid_ids = all_ids[:num_valid_ids]
        self.train_ids = all_ids[num_valid_ids:self.dataset_len-num_test_ids]
        self.test_ids = all_ids[self.dataset_len-num_test_ids:]

    def get_datasets(self, dataset):
        trainset = self.get_train_dataset(dataset)
        valset = self.get_val_dataset(dataset)
        testset = self.get_test_dataset(dataset)
        return trainset, valset, testset

    def get_train_dataset(self, dataset):
        """ Creates a training dataset from existing dataset

        Args:
            dataset (:class:`torch.utils.data.Dataset`): Dataset to be split into a training dataset

        Returns:
            :class:`torch.utils.data.Dataset`: Training dataset split from whole dataset
        """
        return SubsetDataset(dataset, self.train_ids)

    def get_val_dataset(self, dataset):
        """ Creates a validation dataset from existing dataset

        Args:
        dataset (:class:`torch.utils.data.Dataset`): Dataset to be split into a validation dataset

        Returns:
            :class:`torch.utils.data.Dataset`: Validation dataset split from whole dataset
        """
        return SubsetDataset(dataset, self.valid_ids)

    def get_test_dataset(self, dataset):
        """ Creates a test dataset from existing dataset

        Args:
        dataset (:class:`torch.utils.data.Dataset`): Dataset to be split into a validation dataset

        Returns:
            :class:`torch.utils.data.Dataset`: Validation dataset split from whole dataset
        """
        return SubsetDataset(dataset, self.test_ids)


class SubsetDataset(Dataset):
    def __init__(self, dataset, ids):
        """ Dataset that consists of a subset of a previous dataset

        Args:
            dataset (:class:`torch.utils.data.Dataset`): Complete dataset
            ids (list): List of subset IDs
        """
        super(SubsetDataset, self).__init__()
        self.dataset = dataset
        self.ids = ids

    def __getitem__(self, index):
        return self.dataset.__getitem__(self.ids[index])

    def __len__(self):
        return len(self.ids)