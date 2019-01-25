import unittest

from mock import Mock

import torchbearer
from torchbearer.cv_utils import *


class TestCVUtils(unittest.TestCase):

    def test_train_valid_splitter_sizes(self):
        x = range(1, 101)
        y = range(1, 101)

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        valid_split = 0.1
        shuffle = False

        x, y, x_val, y_val = train_valid_splitter(x, y, valid_split, shuffle)
        self.assertTrue(x.size()[0] == 90)
        self.assertTrue(y.size()[0] == 90)
        self.assertTrue(x_val.size()[0] == 10)
        self.assertTrue(y_val.size()[0] == 10)

    def test_train_valid_splitter_sizes_2(self):
        x = range(1, 101)
        y = range(1, 101)

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        valid_split = 0.4
        shuffle = False

        x, y, x_val, y_val = train_valid_splitter(x, y, valid_split, shuffle)
        self.assertTrue(x.size()[0] == 60)
        self.assertTrue(y.size()[0] == 60)
        self.assertTrue(x_val.size()[0] == 40)
        self.assertTrue(y_val.size()[0] == 40)

    def test_train_valid_splitter_sizes_2d(self):
        x = range(1, 101)
        y = range(1, 101)

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        x = torch.stack([x, x], -1)
        y = torch.stack([y, y], -1)

        valid_split = 0.1
        shuffle = False

        x, y, x_val, y_val = train_valid_splitter(x, y, valid_split, shuffle)
        self.assertTrue(list(x.size()) == [90, 2])
        self.assertTrue(list(y.size()) == [90, 2])
        self.assertTrue(list(x_val.size()) == [10, 2])
        self.assertTrue(list(y_val.size()) == [10, 2])

    def test_train_valid_splitter_order(self):
        x = range(1, 101)
        y = range(1, 101)

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        valid_split = 0.1
        shuffle = False

        x, y, x_val, y_val = train_valid_splitter(x, y, valid_split, shuffle)
        self.assertTrue(list(x.numpy()) == list(range(11, 101)))
        self.assertTrue(list(y.numpy()) == list(range(11, 101)))
        self.assertTrue(list(x_val.numpy()) == list(range(1, 11)))
        self.assertTrue(list(y_val.numpy()) == list(range(1, 11)))

    def test_train_valid_splitter_split_negative(self):
        x = range(1, 101)
        y = range(1, 101)

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        valid_split = -0.1
        shuffle = False

        x, y, x_val, y_val = train_valid_splitter(x, y, valid_split, shuffle)
        self.assertTrue(list(x.numpy()) == list(range(91, 101)))
        self.assertTrue(list(y.numpy()) == list(range(91, 101)))
        self.assertTrue(list(x_val.numpy()) == list(range(1, 91)))
        self.assertTrue(list(y_val.numpy()) == list(range(1, 91)))

    def test_train_valid_splitter_split_zero(self):
        x = range(1, 101)
        y = range(1, 101)

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        valid_split = 0
        shuffle = False

        x, y, x_val, y_val = train_valid_splitter(x, y, valid_split, shuffle)
        self.assertTrue(list(x.numpy()) == list(range(1, 101)))
        self.assertTrue(list(y.numpy()) == list(range(1, 101)))
        self.assertTrue(list(x_val.numpy()) == list(range(0, 0)))
        self.assertTrue(list(y_val.numpy()) == list(range(0, 0)))

    def test_train_valid_splitter_split_too_big(self):
        x = range(1, 101)
        y = range(1, 101)

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        valid_split = 1.8
        shuffle = False

        x, y, x_val, y_val = train_valid_splitter(x, y, valid_split, shuffle)
        self.assertTrue(list(x.numpy()) == list(range(0, 0)))
        self.assertTrue(list(y.numpy()) == list(range(0, 0)))
        self.assertTrue(list(x_val.numpy()) == list(range(1, 101)))
        self.assertTrue(list(y_val.numpy()) == list(range(1, 101)))

    def test_train_valid_splitter_shuffle_size(self):
        x = range(1, 101)
        y = range(1, 101)

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        valid_split = 0.1
        shuffle = True

        x, y, x_val, y_val = train_valid_splitter(x, y, valid_split, shuffle)
        self.assertTrue(x.size()[0] == 90)
        self.assertTrue(y.size()[0] == 90)
        self.assertTrue(x_val.size()[0] == 10)
        self.assertTrue(y_val.size()[0] == 10)

    def test_get_train_valid_sets_splitter_args(self):
        x = range(1, 101)
        y = range(1, 101)

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        valid_split = 0.1
        shuffle = True

        torchbearer.cv_utils.train_valid_splitter = Mock(return_value=(x,y,x,y))
        tvs = torchbearer.cv_utils.train_valid_splitter

        trainset, valset = get_train_valid_sets(x, y, None, valid_split, shuffle)
        self.assertEqual(tvs.call_count, 1)
        self.assertTrue(tvs.call_args[0][-1] == valid_split)
        self.assertTrue(list(tvs.call_args[0][0].numpy()) == list(x.numpy()))
        self.assertTrue(list(tvs.call_args[0][1].numpy()) == list(y.numpy()))
        self.assertTrue(tvs.call_args[1]['shuffle'] == shuffle)

    def test_get_train_valid_sets_given_valid_data(self):
        x = range(1, 101)
        y = range(1, 101)
        x_val = range(101, 121)
        y_val = range(101, 121)

        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x_val = torch.Tensor(x_val)
        y_val = torch.Tensor(y_val)

        valid_split = 0.1
        shuffle = False

        trainset, valset = get_train_valid_sets(x, y, (x_val, y_val), valid_split, shuffle)
        self.assertTrue(len(valset) == len(x_val))

    def test_get_train_valid_sets_no_valid(self):
        x = range(1, 101)
        y = range(1, 101)

        x = torch.Tensor(x)
        y = torch.Tensor(y)

        valid_split = None
        shuffle = False

        trainset, valset = get_train_valid_sets(x, y, None, valid_split, shuffle)
        self.assertTrue(valset is None)
        self.assertTrue(len(trainset) == len(x))

    def test_DatasetValidationSplitter(self):
        data = torch.Tensor(list(range(1000)))
        dataset = TensorDataset(data)

        splitter = DatasetValidationSplitter(len(dataset), 0.1)
        trainset = splitter.get_train_dataset(dataset)
        validset = splitter.get_val_dataset(dataset)

        self.assertTrue(len(trainset) == 900)
        self.assertTrue(len(validset) == 100)

        # Check for ids in both train and validation set
        collision = False
        for id in trainset:
            if id in validset.ids:
                collision = True
        self.assertFalse(collision)

    def test_DatasetValidationSplitter_seed(self):
        data = torch.Tensor(list(range(1000)))
        dataset = TensorDataset(data)

        splitter_1 = DatasetValidationSplitter(len(dataset), 0.1, shuffle_seed=1)
        trainset_1 = splitter_1.get_train_dataset(dataset)
        validset_1 = splitter_1.get_val_dataset(dataset)

        splitter_2 = DatasetValidationSplitter(len(dataset), 0.1, shuffle_seed=1)
        trainset_2 = splitter_2.get_train_dataset(dataset)
        validset_2 = splitter_2.get_val_dataset(dataset)

        self.assertTrue(trainset_1.ids[0] == trainset_2.ids[0])
        self.assertTrue(validset_1.ids[0] == validset_2.ids[0])


