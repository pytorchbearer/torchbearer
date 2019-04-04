import torchbearer as tb
from torch.utils.data import Dataset
from unittest import TestCase


class MockDataset(Dataset):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.length


class TestInferSplit(TestCase):
    def test_main_only(self):
        s1, s2, s3 = tb.callbacks.TrainValTestSplit._infer_split(0.8, None, None)
        self.assertAlmostEqual(s1, 0.8, 4)
        self.assertAlmostEqual(s2, 0.2, 4)
        self.assertTrue(s3 is None)

    def test_main_second_only(self):
        s1, s2, s3 = tb.callbacks.TrainValTestSplit._infer_split(0.7, 0.3, None)
        self.assertAlmostEqual(s1, 0.7, 4)
        self.assertAlmostEqual(s2, 0.3, 4)
        self.assertTrue(s3 is None)

    def test_main_second_test(self):
        s1, s2, s3 = tb.callbacks.TrainValTestSplit._infer_split(0.7, 0.2, None)
        self.assertAlmostEqual(s1, 0.7, 4)
        self.assertAlmostEqual(s2, 0.2, 4)
        self.assertAlmostEqual(s3, 0.1, 4)

    def test_main_second_test_defined(self):
        s1, s2, s3 = tb.callbacks.TrainValTestSplit._infer_split(0.7, 0.2, 0.1)
        self.assertAlmostEqual(s1, 0.7, 4)
        self.assertAlmostEqual(s2, 0.2, 4)
        self.assertAlmostEqual(s3, 0.1, 4)

    def test_main_test(self):
        s1, s2, s3 = tb.callbacks.TrainValTestSplit._infer_split(0.7, None, 0.1)
        self.assertAlmostEqual(s1, 0.7, 4)
        self.assertAlmostEqual(s2, 0.2, 4)
        self.assertAlmostEqual(s3, 0.1, 4)


class TestSizeToFloat(TestCase):
    def test_all_float(self):
        ds = MockDataset(100)
        (s1, s2, s3) = tb.callbacks.TrainValTestSplit(ds)._size_to_frac(0.5, 0.2, 0.3)
        self.assertTrue(s1 == 0.5)
        self.assertTrue(s2 == 0.2)
        self.assertTrue(s3 == 0.3)

    def test_all_int(self):
        ds = MockDataset(100)
        (s1, s2, s3) = tb.callbacks.TrainValTestSplit(ds)._size_to_frac(50, 20, 30)
        self.assertTrue(s1 == 0.5)
        self.assertTrue(s2 == 0.2)
        self.assertTrue(s3 == 0.3)

    def test_none(self):
        ds = MockDataset(100)
        (s1, s2, s3) = tb.callbacks.TrainValTestSplit(ds)._size_to_frac(0.8, None, 10)
        self.assertTrue(s1 == 0.8)
        self.assertTrue(s2 is None)
        self.assertTrue(s3 == 0.1)


class TestGetSplitSizes(TestCase):
    def test_all(self):
        ds = MockDataset(100)
        splitter = tb.callbacks.TrainValTestSplit(ds, 0.8, 0.1, 0.1)
        (s1, s2, s3) = splitter._get_split_sizes(0.8, 0.1, 0.1)
        self.assertAlmostEqual(s1, 0.8)
        self.assertAlmostEqual(s2, 0.1)
        self.assertAlmostEqual(s3, 0.1)

    def test_train_only(self):
        ds = MockDataset(100)
        splitter = tb.callbacks.TrainValTestSplit(ds, 0.8)
        (s1, s2, s3) = splitter._get_split_sizes(0.8, None, None)
        self.assertAlmostEqual(s1, 0.8)
        self.assertAlmostEqual(s2, 0.2)
        self.assertAlmostEqual(s3, 0.0)

    def test_train_val(self):
        ds = MockDataset(100)
        splitter = tb.callbacks.TrainValTestSplit(ds, 0.8, 0.2)
        (s1, s2, s3) = splitter._get_split_sizes(0.8, 0.2, None)
        self.assertAlmostEqual(s1, 0.8)
        self.assertAlmostEqual(s2, 0.2)
        self.assertAlmostEqual(s3, 0.0)

    def test_train_val_2(self):
        ds = MockDataset(100)
        splitter = tb.callbacks.TrainValTestSplit(ds, 0.8, 0.1)
        (s1, s2, s3) = splitter._get_split_sizes(0.8, 0.1, None)
        self.assertAlmostEqual(s1, 0.8)
        self.assertAlmostEqual(s2, 0.1)
        self.assertAlmostEqual(s3, 0.1)

    def test_train_test(self):
        ds = MockDataset(100)
        splitter = tb.callbacks.TrainValTestSplit(ds, 0.8, None, 0.1)
        (s1, s2, s3) = splitter._get_split_sizes(0.8, None, 0.1)
        self.assertAlmostEqual(s1, 0.8)
        self.assertAlmostEqual(s2, 0.1)
        self.assertAlmostEqual(s3, 0.1)

    def test_train_test_2(self):
        ds = MockDataset(100)
        splitter = tb.callbacks.TrainValTestSplit(ds, 0.8, None, 0.2)
        (s1, s2, s3) = splitter._get_split_sizes(0.8, None, 0.2)
        self.assertAlmostEqual(s1, 0.8)
        self.assertAlmostEqual(s2, 0.0)
        self.assertAlmostEqual(s3, 0.2)

    def test_val_only(self):
        ds = MockDataset(100)
        splitter = tb.callbacks.TrainValTestSplit(ds, None, 0.2)
        (s1, s2, s3) = splitter._get_split_sizes(None, 0.2, None)
        self.assertAlmostEqual(s1, 0.8)
        self.assertAlmostEqual(s2, 0.2)
        self.assertAlmostEqual(s3, 0.0)

    def test_val_test(self):
        ds = MockDataset(100)
        splitter = tb.callbacks.TrainValTestSplit(ds, None, 0.2, 0.1)
        (s1, s2, s3) = splitter._get_split_sizes(None, 0.2, 0.1)
        self.assertAlmostEqual(s1, 0.7)
        self.assertAlmostEqual(s2, 0.2)
        self.assertAlmostEqual(s3, 0.1)

    def test_val_test_2(self):
        ds = MockDataset(100)
        splitter = tb.callbacks.TrainValTestSplit(ds, None, 0.2, 0.8)
        (s1, s2, s3) = splitter._get_split_sizes(None, 0.2, 0.8)
        self.assertAlmostEqual(s1, 0.0)
        self.assertAlmostEqual(s2, 0.2)
        self.assertAlmostEqual(s3, 0.8)