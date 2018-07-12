from unittest import TestCase
from unittest.mock import MagicMock
import torch
from bink import Model

class TestBink(TestCase):







    def test_load_batch_predict_data(self):
        iterator = iter([torch.Tensor[1], torch.Tensor([2])])
        state = {'device': 'cpu', 'dtype': torch.int}
        Model._load_batch_standard(iterator, state)

    def test_load_batch_predict_list(self):
        iterator = iter([(torch.Tensor[1], 1), (torch.Tensor([2]), 2)])
        state = {'device': 'cpu', 'dtype': torch.int}
        Model._load_batch_standard(iterator, state)