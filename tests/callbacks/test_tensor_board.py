import os

from unittest import TestCase
from unittest.mock import patch, Mock, ANY

import bink
from bink.callbacks import TensorBoard, TensorBoardImages
import torch
import torch.nn as nn


class TestTensorBoard(TestCase):
    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_log_dir(self, mock_board):
        state = {bink.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoard(write_epoch_metrics=False)
        tboard.on_start(state)

        mock_board.assert_called_once_with(log_dir=os.path.join('./logs', 'Sequential_bink'))

    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_batch_log_dir(self, mock_board):
        state = {bink.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), bink.EPOCH: 0}

        tboard = TensorBoard(write_batch_metrics=True, write_graph=False, write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_start_epoch(state)

        mock_board.assert_called_with(log_dir=os.path.join('./logs', 'Sequential_bink', 'epoch-0'))

    @patch('bink.callbacks.tensor_board.SummaryWriter')
    @patch('torch.rand')
    def test_write_graph(self, mock_rand, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_graph = Mock()
        mock_rand.return_value = 1

        state = {bink.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), bink.X: torch.zeros(1, 1, 9, 9)}

        tboard = TensorBoard(write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_sample(state)

        mock_rand.assert_called_once_with(state[bink.X].size(), requires_grad=False)
        mock_board.return_value.add_graph.assert_called_once()
        self.assertEqual(str(state[bink.MODEL]), str(mock_board.return_value.add_graph.call_args_list[0][0][0]))
        self.assertNotEqual(state[bink.MODEL], mock_board.return_value.add_graph.call_args_list[0][0][0])

    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_writer_closed_on_end(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {bink.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoard(write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_end({})
        mock_board.return_value.close.assert_called_once()

    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_batch_writer_closed_on_end_epoch(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {bink.EPOCH: 0}

        tboard = TensorBoard(write_batch_metrics=True, write_epoch_metrics=False)
        tboard.on_start_epoch(state)
        tboard.on_end_epoch({})
        mock_board.return_value.close.assert_called_once()

    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_batch_metrics(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_scalar = Mock()

        state = {bink.EPOCH: 0, bink.METRICS: {'test': 1}, bink.BATCH: 0}

        tboard = TensorBoard(write_batch_metrics=True, write_epoch_metrics=False)
        tboard.on_start_epoch(state)
        tboard.on_step_training(state)
        mock_board.return_value.add_scalar.assert_called_once_with('batch/test', 1, 0)
        mock_board.return_value.add_scalar.reset_mock()
        tboard.on_step_validation(state)
        mock_board.return_value.add_scalar.assert_called_once_with('batch/test', 1, 0)

    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_epoch_metrics(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_scalar = Mock()

        state = {bink.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), bink.EPOCH: 0, bink.METRICS: {'test': 1}}

        tboard = TensorBoard(write_batch_metrics=False, write_epoch_metrics=True)
        tboard.on_start(state)
        tboard.on_end_epoch(state)
        mock_board.return_value.add_scalar.assert_called_once_with('epoch/test', 1, 0)


class TestTensorBoardImages(TestCase):
    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_log_dir(self, mock_board):
        state = {bink.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(log_dir='./test', comment='bink')
        tboard.on_start(state)

        mock_board.assert_called_once_with(log_dir=os.path.join('./test', 'Sequential_bink'))

    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_writer_closed_on_end(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {bink.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages()
        tboard.on_start(state)
        tboard.on_end({})
        mock_board.return_value.close.assert_called_once()

    @patch('bink.callbacks.tensor_board.utils.make_grid')
    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_simple_case(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), bink.EPOCH: 1, bink.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=False, num_images=18, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == state['x'].size())

    @patch('bink.callbacks.tensor_board.utils.make_grid')
    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_multi_batch(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), bink.EPOCH: 1, bink.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=False, num_images=36, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(36, 3, 10, 10).size())

    @patch('bink.callbacks.tensor_board.utils.make_grid')
    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_multi_epoch(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), bink.EPOCH: 1, bink.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=True, num_images=36, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)
        tboard.on_end_epoch(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(36, 3, 10, 10).size())

    @patch('bink.callbacks.tensor_board.utils.make_grid')
    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_single_channel(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 10, 10), bink.EPOCH: 1, bink.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=True, num_images=18, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(18, 1, 10, 10).size())

    @patch('bink.callbacks.tensor_board.utils.make_grid')
    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_odd_batches(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), bink.EPOCH: 1, bink.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=True, num_images=40, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)
        tboard.on_step_validation(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(40, 3, 10, 10).size())
