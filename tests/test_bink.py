from unittest import TestCase
from unittest.mock import patch, Mock

from bink import Model
from bink.bink import _build_bink_state_object
import torch


class TestBink(TestCase):
    @patch('torch.save')
    def test_save_simple(self, mock_save):
        model = Model(None, None, None, [])
        model_file = 'test_model_file.pt'
        state_file = 'test_state_file.bink'

        model.save(model_file, state_file)
        self.assertTrue(mock_save.call_args_list[0][0] == (None, 'test_model_file.pt'))
        self.assertTrue(mock_save.call_args_list[1][0] == ({'optimizer': None, 'criterion': None, 'use_cuda': False},
                                                           'test_state_file.bink'))

    @patch('torch.save')
    def test_save_state_dict(self, mock_save):
        torch_model = torch.nn.Sequential(torch.nn.Conv2d(3,3,3))
        torch_model.state_dict = Mock(return_value=-1)
        model = Model(torch_model, None, None, [])
        model_file = 'test_model_file.pt'
        state_file = 'test_state_file.bink'

        model.save(model_file, state_file, save_state_dict=True)
        self.assertTrue(mock_save.call_args_list[0][0] == (-1, 'test_model_file.pt'))
        self.assertTrue(mock_save.call_args_list[1][0] == ({'optimizer': None, 'criterion': None, 'use_cuda': False},
                                                           'test_state_file.bink'))
    @patch('torch.save')
    def test_save_keys(self, mock_save):
        model = Model(None, None, None, [])
        model_file = 'test_model_file.pt'
        state_file = 'test_state_file.bink'

        model.save(model_file, state_file, save_keys=['criterion'])
        self.assertTrue(mock_save.call_args_list[1][0] == ({'criterion': None}, 'test_state_file.bink'))

    def test_build_state_object(self):
        state = {'model': 1, 'optimiser': 2, 'criterion': 3, 'test': 4}
        built_state = _build_bink_state_object(state, ['test'])
        self.assertDictEqual({'test': 4}, built_state)

        built_state = _build_bink_state_object(state, ['optimiser', 'test'])
        self.assertDictEqual({'optimiser':2, 'test': 4}, built_state)

    # TODO: Test load and restore