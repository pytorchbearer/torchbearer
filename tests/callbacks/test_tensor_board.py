import os

from unittest import TestCase
from unittest.mock import patch, Mock

from bink.callbacks import TensorBoard
import torch
import torch.nn as nn


class TestTensorBoard(TestCase):
    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_log_dir(self, mock_board):
        state = {'model': nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoard(write_epoch_metrics=False)
        tboard.on_start(state)

        mock_board.assert_called_once_with(log_dir=os.path.join('./logs', 'Sequential_bink'))

    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_batch_log_dir(self, mock_board):
        state = {'model': nn.Sequential(nn.Conv2d(3, 3, 3)), 'epoch': 0}

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

        state = {'model': nn.Sequential(nn.Conv2d(3, 3, 3)), 'x': torch.zeros(1, 1, 9, 9)}

        tboard = TensorBoard(write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_sample(state)

        mock_rand.assert_called_once_with(state['x'].size(), requires_grad=False)
        mock_board.return_value.add_graph.assert_called_once()
        self.assertEqual(str(state['model']), str(mock_board.return_value.add_graph.call_args_list[0][0][0]))
        self.assertNotEqual(state['model'], mock_board.return_value.add_graph.call_args_list[0][0][0])

    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_writer_closed_on_end(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {'model': nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoard(write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_end({})
        mock_board.return_value.close.assert_called_once()

    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_batch_writer_closed_on_end_epoch(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {'epoch': 0}

        tboard = TensorBoard(write_batch_metrics=True, write_epoch_metrics=False)
        tboard.on_start_epoch(state)
        tboard.on_end_epoch({})
        mock_board.return_value.close.assert_called_once()

    @patch('bink.callbacks.tensor_board.SummaryWriter')
    def test_batch_metrics(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_scalar = Mock()

        state = {'epoch': 0, 'metrics': {'test': 1}, 't': 0}

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

        state = {'model': nn.Sequential(nn.Conv2d(3, 3, 3)), 'epoch': 0, 'metrics': {'test': 1}}

        tboard = TensorBoard(write_batch_metrics=False, write_epoch_metrics=True)
        tboard.on_start(state)
        tboard.on_end_epoch(state)
        mock_board.return_value.add_scalar.assert_called_once_with('epoch/test', 1, 0)
