import os

from unittest import TestCase
from unittest.mock import patch, Mock, ANY

import torchbearer
from torchbearer.callbacks import TensorBoard, TensorBoardImages, TensorBoardProjector
import torch
import torch.nn as nn


class TestTensorBoard(TestCase):
    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_log_dir(self, mock_board):
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoard(write_epoch_metrics=False)
        tboard.on_start(state)

        mock_board.assert_called_once_with(log_dir=os.path.join('./logs', 'Sequential_torchbearer'))

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_batch_log_dir(self, mock_board):
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.EPOCH: 0}

        tboard = TensorBoard(write_batch_metrics=True, write_graph=False, write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_start_epoch(state)

        mock_board.assert_called_with(log_dir=os.path.join('./logs', 'Sequential_torchbearer', 'epoch-0'))

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    @patch('torch.rand')
    def test_write_graph(self, mock_rand, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_graph = Mock()
        mock_rand.return_value = 1

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.X: torch.zeros(1, 1, 9, 9)}

        tboard = TensorBoard(write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_sample(state)

        mock_rand.assert_called_once_with(state[torchbearer.X].size(), requires_grad=False)
        mock_board.return_value.add_graph.assert_called_once()
        self.assertEqual(str(state[torchbearer.MODEL]), str(mock_board.return_value.add_graph.call_args_list[0][0][0]))
        self.assertNotEqual(state[torchbearer.MODEL], mock_board.return_value.add_graph.call_args_list[0][0][0])

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_writer_closed_on_end(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoard(write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_end({})
        mock_board.return_value.close.assert_called_once()

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_batch_writer_closed_on_end_epoch(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {torchbearer.EPOCH: 0}

        tboard = TensorBoard(write_batch_metrics=True, write_epoch_metrics=False)
        tboard.on_start_epoch(state)
        tboard.on_end_epoch({})
        mock_board.return_value.close.assert_called_once()

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_batch_metrics(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_scalar = Mock()

        state = {torchbearer.EPOCH: 0, torchbearer.METRICS: {'test': 1}, torchbearer.BATCH: 0}

        tboard = TensorBoard(write_batch_metrics=True, write_epoch_metrics=False)
        tboard.on_start_epoch(state)
        tboard.on_step_training(state)
        mock_board.return_value.add_scalar.assert_called_once_with('batch/test', 1, 0)
        mock_board.return_value.add_scalar.reset_mock()
        tboard.on_step_validation(state)
        mock_board.return_value.add_scalar.assert_called_once_with('batch/test', 1, 0)

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_epoch_metrics(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_scalar = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.EPOCH: 0, torchbearer.METRICS: {'test': 1}}

        tboard = TensorBoard(write_batch_metrics=False, write_epoch_metrics=True)
        tboard.on_start(state)
        tboard.on_end_epoch(state)
        mock_board.return_value.add_scalar.assert_called_once_with('epoch/test', 1, 0)


class TestTensorBoardImages(TestCase):
    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_log_dir(self, mock_board):
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(log_dir='./test', comment='torchbearer')
        tboard.on_start(state)

        mock_board.assert_called_once_with(log_dir=os.path.join('./test', 'Sequential_torchbearer'))

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_writer_closed_on_end(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages()
        tboard.on_start(state)
        tboard.on_end({})
        mock_board.return_value.close.assert_called_once()

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_simple_case(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 1, torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=False, num_images=18, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == state['x'].size())

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_multi_batch(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 1, torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=False, num_images=36, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(36, 3, 10, 10).size())

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_multi_epoch(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 1, torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=True, num_images=36, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)
        tboard.on_end_epoch(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(36, 3, 10, 10).size())

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_single_channel(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 10, 10), torchbearer.EPOCH: 1, torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=True, num_images=18, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(18, 1, 10, 10).size())

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_odd_batches(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 1, torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=True, num_images=40, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)
        tboard.on_step_validation(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True, pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(40, 3, 10, 10).size())


class TestTensorBoardProjector(TestCase):
    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_log_dir(self, mock_board):
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardProjector(log_dir='./test', comment='torchbearer')
        tboard.on_start(state)

        mock_board.assert_called_once_with(log_dir=os.path.join('./test', 'Sequential_torchbearer'))

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_writer_closed_on_end(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardProjector()
        tboard.on_start(state)
        tboard.on_end({})
        mock_board.return_value.close.assert_called_once()

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_simple_case(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_embedding = Mock()

        state = {torchbearer.X: torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 0, torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.Y_TRUE: torch.ones(18), torchbearer.BATCH: 0}

        tboard = TensorBoardProjector(num_images=18, avg_data_channels=False, write_data=False, features_key=torchbearer.Y_TRUE)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='features', global_step=0)
        self.assertTrue(mock_board.return_value.add_embedding.call_args[0][0].size() == state[torchbearer.Y_TRUE].unsqueeze(1).size())
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == state[torchbearer.Y_TRUE].size())
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == state[torchbearer.X].size())

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_multi_epoch(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_embedding = Mock()

        state = {torchbearer.X: torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 0, torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.Y_TRUE: torch.ones(18), torchbearer.BATCH: 0}

        tboard = TensorBoardProjector(num_images=18, avg_data_channels=False, write_data=False, features_key=torchbearer.Y_TRUE)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='features', global_step=0)
        self.assertTrue(mock_board.return_value.add_embedding.call_args[0][0].size() == state[torchbearer.Y_TRUE].unsqueeze(1).size())
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == state[torchbearer.Y_TRUE].size())
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == state[torchbearer.X].size())

        tboard.on_end_epoch({})
        mock_board.return_value.add_embedding.reset_mock()

        tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='features',
                                                                      global_step=0)
        self.assertTrue(mock_board.return_value.add_embedding.call_args[0][0].size() == state[torchbearer.Y_TRUE].unsqueeze(1).size())
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == state[torchbearer.Y_TRUE].size())
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == state[torchbearer.X].size())

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_multi_batch(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_embedding = Mock()

        state = {torchbearer.X: torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 0, torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.Y_TRUE: torch.ones(18), torchbearer.BATCH: 0}

        tboard = TensorBoardProjector(num_images=45, avg_data_channels=False, write_data=False, features_key=torchbearer.Y_TRUE)

        tboard.on_start(state)
        for i in range(3):
            state[torchbearer.BATCH] = i
            tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='features', global_step=0)
        self.assertTrue(mock_board.return_value.add_embedding.call_args[0][0].size() == torch.Size([45, 1]))
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == torch.Size([45]))
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == torch.Size([45, 3, 10, 10]))

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_multi_batch_data(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_embedding = Mock()

        state = {torchbearer.X: torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 0, torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.Y_TRUE: torch.ones(18), torchbearer.BATCH: 0}

        tboard = TensorBoardProjector(num_images=45, avg_data_channels=False, write_data=True, write_features=False)

        tboard.on_start(state)
        for i in range(3):
            state[torchbearer.BATCH] = i
            tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='data', global_step=-1)
        self.assertTrue(mock_board.return_value.add_embedding.call_args[0][0].size() == torch.Size([45, 300]))
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == torch.Size([45]))
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == torch.Size([45, 3, 10, 10]))

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_channel_average(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_embedding = Mock()

        state = {torchbearer.X: torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 0, torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.Y_TRUE: torch.ones(18), torchbearer.BATCH: 0}

        tboard = TensorBoardProjector(num_images=18, avg_data_channels=True, write_data=True, write_features=False)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='data', global_step=-1)
        self.assertTrue(mock_board.return_value.add_embedding.call_args[0][0].size() == torch.Size([18, 100]))
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == state[torchbearer.Y_TRUE].size())
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == state[torchbearer.X].size())

    @patch('torchbearer.callbacks.tensor_board.SummaryWriter')
    def test_no_channels(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_embedding = Mock()

        state = {torchbearer.X: torch.ones(18, 10, 10), torchbearer.EPOCH: 0, torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.Y_TRUE: torch.ones(18), torchbearer.BATCH: 0}

        tboard = TensorBoardProjector(num_images=18, avg_data_channels=False, write_data=True, write_features=False)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='data', global_step=-1)
        self.assertTrue(mock_board.return_value.add_embedding.call_args[0][0].size() == torch.Size([18, 100]))
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == state[torchbearer.Y_TRUE].size())
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == torch.Size([18, 1, 10, 10]))
