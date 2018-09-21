import os
from unittest import TestCase
from unittest.mock import patch, Mock, ANY, MagicMock

import torch
import torch.nn as nn

import torchbearer
from torchbearer.callbacks import TensorBoard, TensorBoardImages, TensorBoardProjector, TensorBoardText


class TestTensorBoard(TestCase):
    @patch('tensorboardX.SummaryWriter')
    def test_log_dir(self, mock_board):
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoard(write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_end(state)

        mock_board.assert_called_once_with(log_dir=os.path.join('./logs', 'Sequential_torchbearer'))

    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_log_dir_visdom(self, mock_visdom, mock_writer, _):
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}
        mock_writer.__delete__ = Mock()

        tboard = TensorBoard(visdom=True, write_epoch_metrics=False)

        tboard.on_start(state)
        tboard.on_end(state)

        self.assertEqual(mock_visdom.call_count, 1)
        self.assertTrue(mock_visdom.call_args[1]['log_to_filename'] == os.path.join('./logs', 'Sequential_torchbearer',
                                                                                    'log.log'))

    @patch('tensorboardX.SummaryWriter')
    def test_batch_log_dir(self, mock_board):
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.EPOCH: 0}

        tboard = TensorBoard(write_batch_metrics=True, write_graph=False, write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_start_epoch(state)
        tboard.on_end_epoch(state)
        tboard.on_end(state)

        mock_board.assert_called_with(log_dir=os.path.join('./logs', 'Sequential_torchbearer', 'epoch-0'))

    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_batch_log_dir_visdom(self, mock_visdom,mock_writer, _):
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)),
                 torchbearer.EPOCH: 0, torchbearer.METRICS: {'test': 1}, torchbearer.BATCH: 0}

        tboard = TensorBoard(visdom=True, write_batch_metrics=True, write_graph=False, write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_start_epoch(state)
        tboard.on_end_epoch(state)
        tboard.on_end(state)

        self.assertTrue(mock_visdom.call_args[1]['log_to_filename'] == os.path.join('./logs', 'Sequential_torchbearer', 'epoch', 'log.log'))

    @patch('tensorboardX.SummaryWriter')
    @patch('torch.rand')
    def test_write_graph(self, mock_rand, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_graph = Mock()
        mock_rand.return_value = 1

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.X: torch.zeros(1, 1, 9, 9)}

        tboard = TensorBoard(write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_sample(state)
        tboard.on_end(state)

        mock_rand.assert_called_once_with(state[torchbearer.X].size(), requires_grad=False)
        self.assertEqual(mock_board.return_value.add_graph.call_count, 1)
        self.assertEqual(str(state[torchbearer.MODEL]), str(mock_board.return_value.add_graph.call_args_list[0][0][0]))
        self.assertNotEqual(state[torchbearer.MODEL], mock_board.return_value.add_graph.call_args_list[0][0][0])

    @patch('tensorboardX.SummaryWriter')
    def test_writer_closed_on_end(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoard(write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_end({})
        self.assertEqual(mock_board.return_value.close.call_count, 1)

    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_writer_closed_on_end_visdom(self, mock_visdom, mock_writer, _):
        mock_writer.return_value = Mock()
        mock_writer.return_value.close = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoard(visdom=True, write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_end({})
        self.assertEqual(mock_writer.return_value.close.call_count, 1)

    @patch('tensorboardX.SummaryWriter')
    def test_batch_writer_closed_on_end_epoch(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.EPOCH: 0}

        tboard = TensorBoard(write_batch_metrics=True, write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_start_epoch(state)
        tboard.on_end_epoch({})
        self.assertEqual(mock_board.return_value.close.call_count, 1)
        tboard.on_end(state)

    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_batch_writer_closed_on_end_epoch_visdom(self, mock_visdom, mock_writer, _):
        mock_writer.return_value = Mock()
        mock_writer.return_value.close = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.EPOCH: 0}

        tboard = TensorBoard(visdom=True, write_batch_metrics=True, write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_start_epoch(state)
        tboard.on_end_epoch({})
        tboard.on_end(state)
        self.assertTrue(mock_writer.return_value.close.call_count == 2)

    @patch('tensorboardX.SummaryWriter')
    def test_batch_metrics(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_scalar = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)),
                 torchbearer.EPOCH: 0, torchbearer.METRICS: {'test': 1}, torchbearer.BATCH: 0}

        tboard = TensorBoard(write_batch_metrics=True, write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_start_epoch(state)
        tboard.on_step_training(state)
        mock_board.return_value.add_scalar.assert_called_once_with('batch/test', 1, 0)
        mock_board.return_value.add_scalar.reset_mock()
        tboard.on_step_validation(state)
        mock_board.return_value.add_scalar.assert_called_once_with('batch/test', 1, 0)
        tboard.on_end_epoch(state)
        tboard.on_end(state)

    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_batch_metrics_visdom(self, mock_visdom, mock_writer, _):
        mock_writer.return_value = Mock()
        mock_writer.return_value.add_scalar = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)),
                 torchbearer.EPOCH: 0, torchbearer.METRICS: {'test': 1}, torchbearer.BATCH: 0, torchbearer.TRAIN_STEPS: 0}

        tboard = TensorBoard(visdom=True, write_batch_metrics=True, write_epoch_metrics=False)
        tboard.on_start(state)
        tboard.on_start_epoch(state)
        tboard.on_step_training(state)
        mock_writer.return_value.add_scalar.assert_called_once_with('test', 1, 0, main_tag='batch')
        mock_writer.return_value.add_scalar.reset_mock()
        tboard.on_step_validation(state)
        mock_writer.return_value.add_scalar.assert_called_once_with('test', 1, 0, main_tag='batch')
        tboard.on_end_epoch(state)
        tboard.on_end(state)

    @patch('tensorboardX.SummaryWriter')
    def test_epoch_metrics(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_scalar = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.EPOCH: 0,
                 torchbearer.METRICS: {'test': 1}}

        tboard = TensorBoard(write_batch_metrics=False, write_epoch_metrics=True)
        tboard.on_start(state)
        tboard.on_start_epoch(state)
        tboard.on_end_epoch(state)
        mock_board.return_value.add_scalar.assert_called_once_with('epoch/test', 1, 0)
        tboard.on_end(state)

    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_epoch_metrics_visdom(self, mock_visdom, mock_writer, _):
        mock_writer.return_value = Mock()
        mock_writer.return_value.add_scalar = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.EPOCH: 0,
                 torchbearer.METRICS: {'test': 1}}

        tboard = TensorBoard(visdom=True, write_batch_metrics=False, write_epoch_metrics=True)
        tboard.on_start(state)
        tboard.on_start_epoch(state)
        tboard.on_end_epoch(state)
        mock_writer.return_value.add_scalar.assert_called_once_with('test', 1, 0, main_tag='epoch')
        tboard.on_end(state)


class TestTensorBoardImages(TestCase):
    @patch('tensorboardX.SummaryWriter')
    def test_log_dir(self, mock_board):
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(log_dir='./test', comment='torchbearer')
        tboard.on_start(state)
        tboard.on_end(state)

        mock_board.assert_called_once_with(log_dir=os.path.join('./test', 'Sequential_torchbearer'))

    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_log_dir_visdom(self, mock_visdom, mock_writer, _):
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}
        mock_writer.__delete__ = Mock()

        tboard = TensorBoardImages(visdom=True, log_dir='./test', comment='torchbearer')

        tboard.on_start(state)
        tboard.on_end(state)

        self.assertEqual(mock_visdom.call_count, 1)
        self.assertTrue(mock_visdom.call_args[1]['log_to_filename'] == os.path.join('./test', 'Sequential_torchbearer',
                                                                                    'log.log'))

    @patch('tensorboardX.SummaryWriter')
    def test_writer_closed_on_end(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages()
        tboard.on_start(state)
        tboard.on_end({})
        self.assertEqual(mock_board.return_value.close.call_count, 1)

    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_writer_closed_on_end_visdom_visdom(self, mock_visdom, mock_writer, _):
        mock_writer.return_value = Mock()
        mock_writer.return_value.close = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoard(visdom=True)
        tboard.on_start(state)
        tboard.on_end({})
        self.assertEqual(mock_writer.return_value.close.call_count, 1)

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('tensorboardX.SummaryWriter')
    def test_simple_case(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 1,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=False, num_images=18, nrow=9, padding=3,
                                   normalize=True, norm_range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True,
                                          pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == state['x'].size())
        tboard.on_end({})

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_simple_case_visdom(self, mock_visdom, mock_writer, _, mock_grid):
        mock_writer.return_value = Mock()
        mock_writer.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 1,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(visdom=True, name='test', key='x', write_each_epoch=False, num_images=18, nrow=9, padding=3,
                                   normalize=True, norm_range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True,
                                          pad_value=1)
        mock_writer.return_value.add_image.assert_called_once_with('test1', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == state['x'].size())
        tboard.on_end({})

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('tensorboardX.SummaryWriter')
    def test_multi_batch(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 1,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=False, num_images=36, nrow=9, padding=3,
                                   normalize=True, norm_range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True,
                                          pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(36, 3, 10, 10).size())
        tboard.on_end({})

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_multi_batch_visdom(self, mock_visdom, mock_writer, _, mock_grid):
        mock_writer.return_value = Mock()
        mock_writer.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 1,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(visdom=True, name='test', key='x', write_each_epoch=False, num_images=36, nrow=9, padding=3,
                                   normalize=True, norm_range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True,
                                          pad_value=1)
        mock_writer.return_value.add_image.assert_called_once_with('test1', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(36, 3, 10, 10).size())
        tboard.on_end({})

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('tensorboardX.SummaryWriter')
    def test_multi_epoch(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 1,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=True, num_images=36, nrow=9, padding=3,
                                   normalize=True, norm_range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)
        tboard.on_end_epoch(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True,
                                          pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(36, 3, 10, 10).size())
        tboard.on_end({})

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_multi_epoch_visdom(self, mock_visdom, mock_writer, _, mock_grid):
        mock_writer.return_value = Mock()
        mock_writer.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 1,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(visdom=True, name='test', key='x', write_each_epoch=True, num_images=36, nrow=9, padding=3,
                                   normalize=True, norm_range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)
        tboard.on_end_epoch(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True,
                                          pad_value=1)
        mock_writer.return_value.add_image.assert_called_once_with('test1', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(36, 3, 10, 10).size())
        tboard.on_end({})

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('tensorboardX.SummaryWriter')
    def test_single_channel(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 10, 10), torchbearer.EPOCH: 1,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=True, num_images=18, nrow=9, padding=3,
                                   normalize=True, norm_range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True,
                                          pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(18, 1, 10, 10).size())
        tboard.on_end({})

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_single_channel_visdom(self, mock_visdom, mock_writer, _, mock_grid):
        mock_writer.return_value = Mock()
        mock_writer.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 10, 10), torchbearer.EPOCH: 1,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(visdom=True, name='test', key='x', write_each_epoch=True, num_images=18, nrow=9, padding=3,
                                   normalize=True, norm_range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True,
                                          pad_value=1)
        mock_writer.return_value.add_image.assert_called_once_with('test1', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(18, 1, 10, 10).size())
        tboard.on_end({})

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('tensorboardX.SummaryWriter')
    def test_odd_batches(self, mock_board, mock_grid):
        mock_board.return_value = Mock()
        mock_board.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 1,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(name='test', key='x', write_each_epoch=True, num_images=40, nrow=9, padding=3,
                                   normalize=True, norm_range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)
        tboard.on_step_validation(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True,
                                          pad_value=1)
        mock_board.return_value.add_image.assert_called_once_with('test', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(40, 3, 10, 10).size())
        tboard.on_end({})

    @patch('torchbearer.callbacks.tensor_board.utils.make_grid')
    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_odd_batches_visdom(self, mock_visdom, mock_writer, _, mock_grid):
        mock_writer.return_value = Mock()
        mock_writer.return_value.add_image = Mock()

        mock_grid.return_value = 10

        state = {'x': torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 1,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardImages(visdom=True, name='test', key='x', write_each_epoch=True, num_images=40, nrow=9, padding=3,
                                   normalize=True, norm_range='tmp', scale_each=True, pad_value=1)

        tboard.on_start(state)
        tboard.on_step_validation(state)
        tboard.on_step_validation(state)
        tboard.on_step_validation(state)

        mock_grid.assert_called_once_with(ANY, nrow=9, padding=3, normalize=True, range='tmp', scale_each=True,
                                          pad_value=1)
        mock_writer.return_value.add_image.assert_called_once_with('test1', 10, 1)
        self.assertTrue(mock_grid.call_args[0][0].size() == torch.ones(40, 3, 10, 10).size())
        tboard.on_end({})


class TestTensorBoardProjector(TestCase):
    @patch('tensorboardX.SummaryWriter')
    def test_log_dir(self, mock_board):
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardProjector(log_dir='./test', comment='torchbearer')
        tboard.on_start(state)
        tboard.on_end(state)

        mock_board.assert_called_once_with(log_dir=os.path.join('./test', 'Sequential_torchbearer'))

    @patch('tensorboardX.SummaryWriter')
    def test_writer_closed_on_end(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.close = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3))}

        tboard = TensorBoardProjector()
        tboard.on_start(state)
        tboard.on_end({})
        self.assertEqual(mock_board.return_value.close.call_count, 1)

    @patch('tensorboardX.SummaryWriter')
    def test_simple_case(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_embedding = Mock()

        state = {torchbearer.X: torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 0,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.Y_TRUE: torch.ones(18),
                 torchbearer.BATCH: 0}

        tboard = TensorBoardProjector(num_images=18, avg_data_channels=False, write_data=False,
                                      features_key=torchbearer.Y_TRUE)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='features',
                                                                      global_step=0)
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[0][0].size() == state[torchbearer.Y_TRUE].unsqueeze(
                1).size())
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == state[torchbearer.Y_TRUE].size())
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == state[torchbearer.X].size())
        tboard.on_end(state)

    @patch('tensorboardX.SummaryWriter')
    def test_multi_epoch(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_embedding = Mock()

        state = {torchbearer.X: torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 0,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.Y_TRUE: torch.ones(18),
                 torchbearer.BATCH: 0}

        tboard = TensorBoardProjector(num_images=18, avg_data_channels=False, write_data=False,
                                      features_key=torchbearer.Y_TRUE)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='features',
                                                                      global_step=0)
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[0][0].size() == state[torchbearer.Y_TRUE].unsqueeze(
                1).size())
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == state[torchbearer.Y_TRUE].size())
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == state[torchbearer.X].size())

        tboard.on_end_epoch({})
        mock_board.return_value.add_embedding.reset_mock()

        tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='features',
                                                                      global_step=0)
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[0][0].size() == state[torchbearer.Y_TRUE].unsqueeze(
                1).size())
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == state[torchbearer.Y_TRUE].size())
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == state[torchbearer.X].size())
        tboard.on_end({})

    @patch('tensorboardX.SummaryWriter')
    def test_multi_batch(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_embedding = Mock()

        state = {torchbearer.X: torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 0,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.Y_TRUE: torch.ones(18),
                 torchbearer.BATCH: 0}

        tboard = TensorBoardProjector(num_images=45, avg_data_channels=False, write_data=False,
                                      features_key=torchbearer.Y_TRUE)

        tboard.on_start(state)
        for i in range(3):
            state[torchbearer.BATCH] = i
            tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='features',
                                                                      global_step=0)
        self.assertTrue(mock_board.return_value.add_embedding.call_args[0][0].size() == torch.Size([45, 1]))
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == torch.Size([45]))
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == torch.Size([45, 3, 10, 10]))
        tboard.on_end({})

    @patch('tensorboardX.SummaryWriter')
    def test_multi_batch_data(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_embedding = Mock()

        state = {torchbearer.X: torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 0,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.Y_TRUE: torch.ones(18),
                 torchbearer.BATCH: 0}

        tboard = TensorBoardProjector(num_images=45, avg_data_channels=False, write_data=True, write_features=False)

        tboard.on_start(state)
        for i in range(3):
            state[torchbearer.BATCH] = i
            tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='data',
                                                                      global_step=-1)
        self.assertTrue(mock_board.return_value.add_embedding.call_args[0][0].size() == torch.Size([45, 300]))
        self.assertTrue(mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == torch.Size([45]))
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == torch.Size([45, 3, 10, 10]))
        tboard.on_end({})

    @patch('tensorboardX.SummaryWriter')
    def test_channel_average(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_embedding = Mock()

        state = {torchbearer.X: torch.ones(18, 3, 10, 10), torchbearer.EPOCH: 0,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.Y_TRUE: torch.ones(18),
                 torchbearer.BATCH: 0}

        tboard = TensorBoardProjector(num_images=18, avg_data_channels=True, write_data=True, write_features=False)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='data',
                                                                      global_step=-1)
        self.assertTrue(mock_board.return_value.add_embedding.call_args[0][0].size() == torch.Size([18, 100]))
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == state[torchbearer.Y_TRUE].size())
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == state[torchbearer.X].size())
        tboard.on_end({})

    @patch('tensorboardX.SummaryWriter')
    def test_no_channels(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_embedding = Mock()

        state = {torchbearer.X: torch.ones(18, 10, 10), torchbearer.EPOCH: 0,
                 torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)), torchbearer.Y_TRUE: torch.ones(18),
                 torchbearer.BATCH: 0}

        tboard = TensorBoardProjector(num_images=18, avg_data_channels=False, write_data=True, write_features=False)

        tboard.on_start(state)
        tboard.on_step_validation(state)

        mock_board.return_value.add_embedding.assert_called_once_with(ANY, metadata=ANY, label_img=ANY, tag='data',
                                                                      global_step=-1)
        self.assertTrue(mock_board.return_value.add_embedding.call_args[0][0].size() == torch.Size([18, 100]))
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[1]['metadata'].size() == state[torchbearer.Y_TRUE].size())
        self.assertTrue(
            mock_board.return_value.add_embedding.call_args[1]['label_img'].size() == torch.Size([18, 1, 10, 10]))
        tboard.on_end({})


class TestTensorbardText(TestCase):
    def test_table_formatter_one_metric(self):
        tf = TensorBoardText.table_formatter

        metrics = str({'test_metric_1': 1})
        table = tf(metrics).replace(" ", "")

        correct_table = '<table><th>Metric</th><th>Value</th><tr><td>test_metric_1</td><td>1</td></tr></table>'
        self.assertEqual(table, correct_table)

    def test_table_formatter_two_metrics(self):
        tf = TensorBoardText.table_formatter

        metrics = str({'test_metric_1': 1, 'test_metric_2': 2})
        table = tf(metrics).replace(" ", "")

        self.assertIn('<tr><td>test_metric_1</td><td>1</td></tr>', table)
        self.assertIn('<tr><td>test_metric_2</td><td>2</td></tr>', table)
        
    @patch('tensorboardX.SummaryWriter')
    def test_epoch_writer(self, mock_writer):
        tboard = TensorBoardText(log_trial_summary=False)

        metrics = {'test_metric_1': 1, 'test_metric_2': 1}
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)),
                 torchbearer.EPOCH: 1, torchbearer.METRICS: metrics}
        metric_string = TensorBoardText.table_formatter(str(metrics))

        tboard.on_start(state)
        tboard.on_start_training(state)
        tboard.on_start_epoch(state)
        tboard.on_end_epoch(state)
        mock_writer.return_value.add_text.assert_called_once_with('epoch', metric_string, 1)
        tboard.on_end(state)

    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_epoch_writer_visdom(self, mock_visdom, mock_writer):
        tboard = TensorBoardText(visdom=True, log_trial_summary=False)

        metrics = {'test_metric_1': 1, 'test_metric_2': 1}
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)),
                 torchbearer.EPOCH: 1, torchbearer.METRICS: metrics}
        metric_string = TensorBoardText.table_formatter(str(metrics))

        tboard.on_start(state)
        tboard.on_start_training(state)
        tboard.on_start_epoch(state)
        tboard.on_end_epoch(state)
        mock_writer.return_value.add_text.assert_called_once_with('epoch', '<h4>Epoch 1</h4>'+metric_string, 1)
        tboard.on_end(state)

    @patch('tensorboardX.SummaryWriter')
    def test_batch_writer(self, mock_writer):
        tboard = TensorBoardText(write_epoch_metrics=False, write_batch_metrics=True, log_trial_summary=False)

        metrics = {'test_metric_1': 1, 'test_metric_2': 1}
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)),
                 torchbearer.EPOCH: 1, torchbearer.BATCH: 100, torchbearer.METRICS: metrics}
        metric_string = TensorBoardText.table_formatter(str(metrics))

        tboard.on_start(state)
        tboard.on_start_training(state)
        tboard.on_start_epoch(state)
        tboard.on_step_training(state)
        mock_writer.return_value.add_text.assert_called_once_with('batch', metric_string, 100)
        tboard.on_end_epoch(state)
        tboard.on_end(state)

    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_batch_writer_visdom(self, mock_visdom, mock_writer):
        tboard = TensorBoardText(visdom=True, write_epoch_metrics=False, write_batch_metrics=True, log_trial_summary=False)

        metrics = {'test_metric_1': 1, 'test_metric_2': 1}
        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)),
                 torchbearer.EPOCH: 1, torchbearer.BATCH: 100, torchbearer.METRICS: metrics}
        metric_string = TensorBoardText.table_formatter(str(metrics))
        metric_string = '<h3>Epoch {} - Batch {}</h3>'.format(state[torchbearer.EPOCH], state[torchbearer.BATCH])+metric_string

        tboard.on_start(state)
        tboard.on_start_training(state)
        tboard.on_start_epoch(state)
        tboard.on_step_training(state)
        mock_writer.return_value.add_text.assert_called_once_with('batch', metric_string, 1)
        tboard.on_end_epoch(state)
        tboard.on_end(state)

    @patch('tensorboardX.SummaryWriter')
    def test_batch_metrics(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_text = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)),
                 torchbearer.EPOCH: 0, torchbearer.METRICS: {'test': 1}, torchbearer.BATCH: 0}

        tboard = TensorBoardText(write_batch_metrics=True, write_epoch_metrics=False, log_trial_summary=False)
        tboard.on_start(state)
        tboard.on_start_epoch(state)
        tboard.on_step_training(state)
        mock_board.return_value.add_text.assert_called_once_with('batch', TensorBoardText.table_formatter(str(state[torchbearer.METRICS])), 0)
        mock_board.return_value.add_text.reset_mock()
        tboard.on_end_epoch(state)
        tboard.on_end(state)

    @patch('os.makedirs')
    @patch('tensorboardX.torchvis.VisdomWriter')
    @patch('visdom.Visdom')
    def test_batch_metrics_visdom(self, mock_visdom, mock_writer, _):
        mock_writer.return_value = Mock()
        mock_writer.return_value.add_text = Mock()

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)),
                 torchbearer.EPOCH: 0, torchbearer.METRICS: {'test': 1}, torchbearer.BATCH: 0, torchbearer.TRAIN_STEPS: 0}

        tboard = TensorBoardText(visdom=True, write_batch_metrics=True, write_epoch_metrics=False, log_trial_summary=False)
        tboard.on_start(state)
        tboard.on_start_epoch(state)
        tboard.on_step_training(state)
        mock_writer.return_value.add_text.assert_called_once_with('batch', '<h3>Epoch {} - Batch {}</h3>'.format(state[torchbearer.EPOCH], state[torchbearer.BATCH])+TensorBoardText.table_formatter(str(state[torchbearer.METRICS])), 1)
        mock_writer.return_value.add_text.reset_mock()
        tboard.on_step_validation(state)
        tboard.on_end(state)

    @patch('tensorboardX.SummaryWriter')
    def test_log_summary(self, mock_board):
        mock_board.return_value = Mock()
        mock_board.return_value.add_text = Mock()
        mock_self = 'test'

        state = {torchbearer.MODEL: nn.Sequential(nn.Conv2d(3, 3, 3)),
                 torchbearer.EPOCH: 0, torchbearer.METRICS: {'test': 1}, torchbearer.BATCH: 0, torchbearer.SELF: mock_self}
        tboard = TensorBoardText(write_batch_metrics=False, write_epoch_metrics=False, log_trial_summary=True)
        tboard.on_start(state)
        self.assertEqual(mock_board.return_value.add_text.call_args[0][0], 'trial')
        self.assertEqual(mock_board.return_value.add_text.call_args[0][1], str(mock_self))
