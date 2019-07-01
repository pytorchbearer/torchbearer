from unittest import TestCase
from mock import patch, Mock

import torchbearer
from torchbearer import Trial
from torchbearer.callbacks.checkpointers import _Checkpointer, ModelCheckpoint, MostRecent, Interval, Best
import warnings

class TestCheckpointer(TestCase):
    @patch('os.makedirs')
    def test_make_dirs(self, mock_dirs):
        _Checkpointer('thisdirectoryshouldntexist/norshouldthis/model.pt')
        mock_dirs.assert_called_once_with('thisdirectoryshouldntexist/norshouldthis')

    @patch('torch.save')
    @patch('os.makedirs')
    def test_no_existing_file(self, mock_dirs, mock_save):
        check = _Checkpointer('thisdirectoryshouldntexist/norshouldthis/model.pt')
        check.most_recent = 'thisfiledoesnotexist.pt'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            check.save_checkpoint({torchbearer.METRICS: {}, torchbearer.SELF: Mock()}, True)
            self.assertTrue(len(w) == 1)
            self.assertTrue('Failed to delete old file' in str(w[-1].message))

    @patch("torch.save")
    def test_save_checkpoint_save_filename(self, mock_save):
        torchmodel = Mock()
        optim = Mock()
        state = {
            torchbearer.SELF: Trial(torchmodel, optim, None, []),
            torchbearer.METRICS: {}
        }

        file_format = 'test_file.pt'
        check = _Checkpointer(file_format)
        check.save_checkpoint(state)
        self.assertEqual(mock_save.call_count, 1)

        self.assertTrue(mock_save.call_args[0][1] == 'test_file.pt')

    @patch("torch.save")
    def test_save_checkpoint_formatting(self, mock_save):
        torchmodel = Mock()
        optim = Mock()
        state = {
            torchbearer.SELF: Trial(torchmodel, optim, None, []),
            torchbearer.METRICS: {},
            torchbearer.EPOCH: 2
        }

        file_format = 'test_file_{epoch}.pt'
        check = _Checkpointer(file_format)
        check.save_checkpoint(state)
        self.assertEqual(mock_save.call_count, 1)

        self.assertTrue(mock_save.call_args[0][1] == 'test_file_2.pt')

    @patch("torch.save")
    def test_save_checkpoint_formatting_metric(self, mock_save):
        torchmodel = Mock()
        optim = Mock()
        state = {
            torchbearer.SELF: Trial(torchmodel, optim, None, []),
            torchbearer.METRICS: {'test_metric': 0.001},
            torchbearer.EPOCH: 2
        }

        file_format = 'test_file_{test_metric}.pt'
        check = _Checkpointer(file_format)
        check.save_checkpoint(state)
        self.assertEqual(mock_save.call_count, 1)

        self.assertTrue(mock_save.call_args[0][1] == 'test_file_0.001.pt')

    @patch("torch.save")
    def test_save_checkpoint_subformatting(self, mock_save):
        torchmodel = Mock()
        optim = Mock()
        state = {
            torchbearer.SELF: Trial(torchmodel, optim, None, []),
            torchbearer.METRICS: {'test_metric': 0.001},
            torchbearer.EPOCH: 2
        }

        file_format = 'test_file_{test_metric:.01f}.pt'
        check = _Checkpointer(file_format)
        check.save_checkpoint(state)
        self.assertEqual(mock_save.call_count, 1)

        self.assertTrue(mock_save.call_args[0][1] == 'test_file_0.0.pt')

    @patch("torch.save")
    def test_save_checkpoint_model_only(self, mock_save):
        torchmodel = Mock()
        optim = Mock()
        state = {
            torchbearer.SELF: Trial(torchmodel, optim, None, []),
            torchbearer.METRICS: {'test_metric': 0.001},
            torchbearer.EPOCH: 2,
            torchbearer.MODEL: torchmodel,
        }

        file_format = 'test_file_{test_metric:.01f}.pt'
        check = _Checkpointer(file_format, save_model_params_only=True)
        check.save_checkpoint(state)
        self.assertEqual(mock_save.call_count, 1)
        self.assertTrue(mock_save.call_args[0][0] == torchmodel.state_dict())
        self.assertTrue(mock_save.call_args[0][1] == 'test_file_0.0.pt')


    @patch("torch.save")
    def test_save_checkpoint_wrong_format(self, _):
        torchmodel = Mock()
        optim = Mock()
        state = {
            torchbearer.SELF: Trial(torchmodel, optim, None, []),
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
            torchbearer.SELF: Trial(torchmodel, optim, None, []),
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

        check.on_checkpoint(state)
        check.on_checkpoint(state)

        self.assertTrue(mock_save_check.call_count == 2)


# TODO: Negative and fractional interval test and decide how to handle
class TestInterval(TestCase):
    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_interval_is_1(self, mock_save_check):
        state = {}
        check = Interval('test_file', period=1)

        check.on_checkpoint(state)
        check.on_checkpoint(state)

        self.assertTrue(mock_save_check.call_count == 2)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_interval_is_more_than_1(self, mock_save_check):
        state = {}
        check = Interval('test_file', period=4)

        for i in range(13):
            check.on_checkpoint(state)
            if i == 3:
                self.assertTrue(mock_save_check.call_count == 1)
            elif i == 6:
                self.assertFalse(mock_save_check.call_count == 2)
            elif i == 7:
                self.assertTrue(mock_save_check.call_count == 2)

        self.assertTrue(mock_save_check.call_count == 3)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_interval_on_batch(self, mock_save_check):
        state = {}
        check = Interval('test_file', period=4, on_batch=True)

        for i in range(13):
            check.on_step_training(state)
            if i == 3:
                self.assertTrue(mock_save_check.call_count == 1)
            elif i == 6:
                self.assertFalse(mock_save_check.call_count == 2)
            elif i == 7:
                self.assertTrue(mock_save_check.call_count == 2)
        check.on_checkpoint(state)
        self.assertTrue(mock_save_check.call_count == 3)

    def test_state_dict(self):
        check = Interval('test')
        check.epochs_since_last_save = 10

        state = check.state_dict()

        check = Interval('test')
        check.load_state_dict(state)

        self.assertEqual(check.epochs_since_last_save, 10)


class TestBest(TestCase):
    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_min_with_increasing(self, mock_save):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, mode='min')
        check.on_start(state)

        check.on_checkpoint(state)
        self.assertTrue(mock_save.call_count == 1)

        state = {torchbearer.METRICS: {'val_loss': 0.2}}
        check.on_checkpoint(state)
        self.assertTrue(mock_save.call_count == 1)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_min_with_decreasing(self, mock_save):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, mode='min')
        check.on_start(state)

        check.on_checkpoint(state)
        self.assertTrue(mock_save.call_count == 1)

        state = {torchbearer.METRICS: {'val_loss': 0.001}}
        check.on_checkpoint(state)
        self.assertTrue(mock_save.call_count == 2)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_max_with_increasing(self, mock_save):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, mode='max')
        check.on_start(state)

        check.on_checkpoint(state)
        self.assertTrue(mock_save.call_count == 1)

        state = {torchbearer.METRICS: {'val_loss': 0.2}}
        check.on_checkpoint(state)
        self.assertTrue(mock_save.call_count == 2)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_max_with_decreasing(self, mock_save):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, mode='max')
        check.on_start(state)

        check.on_checkpoint(state)
        self.assertTrue(mock_save.call_count == 1)

        state = {torchbearer.METRICS: {'val_loss': 0.001}}
        check.on_checkpoint(state)
        self.assertTrue(mock_save.call_count == 1)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_min_delta_no_save(self, mock_save):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, mode='min', min_delta=0.1)
        check.on_start(state)

        check.on_checkpoint(state)
        self.assertTrue(mock_save.call_count == 1)

        state = {torchbearer.METRICS: {'val_loss': 0.001}}
        check.on_checkpoint(state)
        self.assertTrue(mock_save.call_count == 1)

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_min_delta_save(self, mock_save):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, mode='min', min_delta=0.1)
        check.on_start(state)

        check.on_checkpoint(state)
        self.assertTrue(mock_save.call_count == 1)

        state = {torchbearer.METRICS: {'val_loss': -0.001}}
        check.on_checkpoint(state)
        self.assertTrue(mock_save.call_count == 2)
        
    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_auto_shoud_be_min(self, _):
        state = {torchbearer.METRICS: {'val_loss': 0.1}}

        file_path = 'test_file_{val_loss:.2f}'
        check = Best(file_path, monitor='val_loss')
        check.on_start(state)

        check.on_checkpoint(state)
        self.assertTrue(check.mode == 'min')

    @patch('torchbearer.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_auto_shoud_be_max(self, _):
        state = {torchbearer.METRICS: {'acc_loss': 0.1}}

        file_path = 'test_file_{acc_loss:.2f}'
        check = Best(file_path, monitor='acc_loss')
        check.on_start(state)

        check.on_checkpoint(state)
        self.assertTrue(check.mode == 'max')

    def test_state_dict(self):
        check = Best('test')
        check.best = 'temp2'
        check.epochs_since_last_save = 10

        state = check.state_dict()

        check = Best('test')
        check.load_state_dict(state)

        self.assertEqual(check.best, 'temp2')
        self.assertEqual(check.epochs_since_last_save, 10)
