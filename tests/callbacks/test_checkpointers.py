from unittest import TestCase
from unittest.mock import patch, Mock

import torchbearer
from torchbearer import Model
from torchbearer.callbacks.checkpointers import _Checkpointer, ModelCheckpoint, MostRecent, Interval, Best


class TestCheckpointer(TestCase):
    @patch('os.makedirs')
    def test_make_dirs(self, mock_dirs):
        _Checkpointer('thisdirectoryshouldntexist/norshouldthis/model.pt')
        mock_dirs.assert_called_once_with('thisdirectoryshouldntexist/norshouldthis')

    @patch("torch.save")
    def test_save_checkpoint_save_filename(self, mock_save):
        torchmodel = Mock()
        optim = Mock()
        state = {
            torchbearer.SELF: Model(torchmodel, optim, None, []),
            torchbearer.METRICS: {}
        }

        file_format = 'test_file.pt'
        check = _Checkpointer(file_format)
        check.save_checkpoint(state)
        mock_save.assert_called_once()

        self.assertTrue(mock_save.call_args[0][1] == 'test_file.pt')

    @patch("torch.save")
    def test_save_checkpoint_formatting(self, mock_save):
        torchmodel = Mock()
        optim = Mock()
        state = {
            torchbearer.SELF: Model(torchmodel, optim, None, []),
            torchbearer.METRICS: {},
            torchbearer.EPOCH: 2
        }

        file_format = 'test_file_{epoch}.pt'
        check = _Checkpointer(file_format)
        check.save_checkpoint(state)
        mock_save.assert_called_once()

        self.assertTrue(mock_save.call_args[0][1] == 'test_file_2.pt')

    @patch("torch.save")
    def test_save_checkpoint_formatting_metric(self, mock_save):
        torchmodel = Mock()
        optim = Mock()
        state = {
            torchbearer.SELF: Model(torchmodel, optim, None, []),
            torchbearer.METRICS: {'test_metric': 0.001},
            torchbearer.EPOCH: 2
        }

        file_format = 'test_file_{test_metric}.pt'
        check = _Checkpointer(file_format)
        check.save_checkpoint(state)
        mock_save.assert_called_once()

        self.assertTrue(mock_save.call_args[0][1] == 'test_file_0.001.pt')

    @patch("torch.save")
    def test_save_checkpoint_subformatting(self, mock_save):
        torchmodel = Mock()
        optim = Mock()
        state = {
            torchbearer.SELF: Model(torchmodel, optim, None, []),
            torchbearer.METRICS: {'test_metric': 0.001},
            torchbearer.EPOCH: 2
        }

        file_format = 'test_file_{test_metric:.01f}.pt'
        check = _Checkpointer(file_format)
        check.save_checkpoint(state)
        mock_save.assert_called_once()

        self.assertTrue(mock_save.call_args[0][1] == 'test_file_0.0.pt')

    @patch("torch.save")
    def test_save_checkpoint_wrong_format(self, _):
        torchmodel = Mock()
        optim = Mock()
        state = {
            torchbearer.SELF: Model(torchmodel, optim, None, []),
            torchbearer.METRICS: {'test_metric': 0.001},
            torchbearer.EPOCH: 2
        }

        file_format = 'test_file_{test_metric:d}.pt'
        check = _Checkpointer(file_format)
        try:
            check.save_checkpoint(state)
        except:
            return

        self.fail('No error was thrown when wrong format chosen for save file format')

    @patch('os.remove')
    @patch("torch.save")
    def test_save_checkpoint_overwrite_recent(self, _, __):
        torchmodel = Mock()
        optim = Mock()
        state = {
            torchbearer.SELF: Model(torchmodel, optim, None, []),
            torchbearer.EPOCH: 0,
            torchbearer.METRICS: {}
        }

        file_format = 'test_file_{epoch}.pt'
        check = _Checkpointer(file_format)
        check.save_checkpoint(state, True)
        self.assertTrue(check.most_recent == 'test_file_0.pt')

        state[torchbearer.EPOCH] = 1
        check.save_checkpoint(state, True)
        self.assertTrue(check.most_recent == 'test_file_1.pt')


class TestModelCheckpoint(TestCase):
    def test_best_only(self):
        self.assertTrue(isinstance(ModelCheckpoint(save_best_only=True), Best))

    def test_not_best_only(self):
        self.assertTrue(isinstance(ModelCheckpoint(save_best_only=False), Interval))


class TestMostRecent(TestCase):
    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_save(self, mock_save_check):
        state = {}
        check = MostRecent('test_file.pt')

        check.on_end_epoch(state)
        check.on_end_epoch(state)

        self.assertTrue(mock_save_check.call_count == 2)


# TODO: Negative and fractional interval test and decide how to handle
class TestInterval(TestCase):
    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_interval_is_1(self, mock_save_check):
        state = {}
        check = Interval('test_file', period=1)

        check.on_end_epoch(state)
        check.on_end_epoch(state)

        self.assertTrue(mock_save_check.call_count == 2)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_interval_is_more_than_1(self, mock_save_check):
        state = {}
        check = Interval('test_file', period=4)

        for i in range(13):
            check.on_end_epoch(state)
            if i == 3:
                self.assertTrue(mock_save_check.call_count == 1)
            elif i == 6:
                self.assertFalse(mock_save_check.call_count == 2)
            elif i == 7:
                self.assertTrue(mock_save_check.call_count == 2)

        self.assertTrue(mock_save_check.call_count == 3)


class TestBest(TestCase):
    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_min_with_increasing(self, mock_save):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, mode='min')
        check.on_start(state)

        check.on_end_epoch(state)
        self.assertTrue(mock_save.call_count == 1)

        state = {torchbearer.METRICS: {'val_loss': 0.2}}
        check.on_end_epoch(state)
        self.assertTrue(mock_save.call_count == 1)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_min_with_decreasing(self, mock_save):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, mode='min')
        check.on_start(state)

        check.on_end_epoch(state)
        self.assertTrue(mock_save.call_count == 1)

        state = {torchbearer.METRICS: {'val_loss': 0.001}}
        check.on_end_epoch(state)
        self.assertTrue(mock_save.call_count == 2)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_max_with_increasing(self, mock_save):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, mode='max')
        check.on_start(state)

        check.on_end_epoch(state)
        self.assertTrue(mock_save.call_count == 1)

        state = {torchbearer.METRICS: {'val_loss': 0.2}}
        check.on_end_epoch(state)
        self.assertTrue(mock_save.call_count == 2)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_max_with_decreasing(self, mock_save):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, mode='max')
        check.on_start(state)

        check.on_end_epoch(state)
        self.assertTrue(mock_save.call_count == 1)

        state = {torchbearer.METRICS: {'val_loss': 0.001}}
        check.on_end_epoch(state)
        self.assertTrue(mock_save.call_count == 1)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_min_delta_no_save(self, mock_save):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, mode='min', min_delta=0.1)
        check.on_start(state)

        check.on_end_epoch(state)
        self.assertTrue(mock_save.call_count == 1)

        state = {torchbearer.METRICS: {'val_loss': 0.001}}
        check.on_end_epoch(state)
        self.assertTrue(mock_save.call_count == 1)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_min_delta_save(self, mock_save):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, mode='min', min_delta=0.1)
        check.on_start(state)

        check.on_end_epoch(state)
        self.assertTrue(mock_save.call_count == 1)

        state = {torchbearer.METRICS: {'val_loss': -0.001}}
        check.on_end_epoch(state)
        self.assertTrue(mock_save.call_count == 2)
        
    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_auto_shoud_be_min(self, _):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, monitor='val_loss')
        check.on_start(state)

        check.on_end_epoch(state)
        self.assertTrue(check.mode == 'min')

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_auto_shoud_be_max(self, _):
        state = {torchbearer.METRICS: {'acc_loss': 0.1}}

        file_path = 'test_file_{acc_loss:.2f}'
        check = Best(file_path, monitor='acc_loss')
        check.on_start(state)

        check.on_end_epoch(state)
        self.assertTrue(check.mode == 'max')
