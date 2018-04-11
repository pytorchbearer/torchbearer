from unittest import TestCase
from unittest.mock import patch, Mock
from bink.callbacks.checkpointers import _Checkpointer, MostRecent, Interval, Best
from bink import Model
import numpy as np


class Test_Checkpointer(TestCase):
    @patch("bink.Model.save")
    def test_save_checkpoint_save_filename(self, mock_save):
        state = {
            'self': Model(None, None, None, []),
            'metrics': {}
        }

        file_format = 'test_file'
        check = _Checkpointer(file_format, False)
        check.save_checkpoint(state)
        mock_save.assert_called_once()

        self.assertTrue(mock_save.call_args_list[0][0][0] == 'test_file.pt')
        self.assertTrue(mock_save.call_args_list[0][0][1] == 'test_file.bink')

    @patch("bink.Model.save")
    def test_save_checkpoint_save_weights(self, mock_save):
        state = {
            'self': Model(None, None, None, []),
            'metrics': {}
        }

        file_format = 'test_file'
        check = _Checkpointer(file_format, False)
        check.save_checkpoint(state)

        self.assertTrue(mock_save.call_args_list[0][0][-1] == False)

        check = _Checkpointer(file_format, True)
        check.save_checkpoint(state)

        self.assertTrue(mock_save.call_args_list[1][0][-1] == True)


    @patch("bink.Model.save")
    def test_save_checkpoint_formatting(self, mock_save):
        state = {
            'self': Model(None, None, None, []),
            'metrics': {},
            'epoch':2
        }

        file_format = 'test_file_{epoch}'
        check = _Checkpointer(file_format, False)
        check.save_checkpoint(state)
        mock_save.assert_called_once()

        self.assertTrue(mock_save.call_args_list[0][0][0] == 'test_file_2.pt')
        self.assertTrue(mock_save.call_args_list[0][0][1] == 'test_file_2.bink')

    @patch("bink.Model.save")
    def test_save_checkpoint_formatting_metric(self, mock_save):
        state = {
            'self': Model(None, None, None, []),
            'metrics': {'test_metric':0.001},
            'epoch':2
        }

        file_format = 'test_file_{test_metric}'
        check = _Checkpointer(file_format, False)
        check.save_checkpoint(state)
        mock_save.assert_called_once()

        self.assertTrue(mock_save.call_args_list[0][0][0] == 'test_file_0.001.pt')
        self.assertTrue(mock_save.call_args_list[0][0][1] == 'test_file_0.001.bink')

    @patch("bink.Model.save")
    def test_save_checkpoint_subformatting(self, mock_save):
        state = {
            'self': Model(None, None, None, []),
            'metrics': {'test_metric':0.001},
            'epoch':2
        }

        file_format = 'test_file_{test_metric:.01f}'
        check = _Checkpointer(file_format, False)
        check.save_checkpoint(state)
        mock_save.assert_called_once()

        self.assertTrue(mock_save.call_args_list[0][0][0] == 'test_file_0.0.pt')
        self.assertTrue(mock_save.call_args_list[0][0][1] == 'test_file_0.0.bink')

    @patch("bink.Model.save")
    def test_save_checkpoint_wrong_format(self, mock_save):
        state = {
            'self': Model(None, None, None, []),
            'metrics': {'test_metric':0.001},
            'epoch':2
        }

        file_format = 'test_file_{test_metric:d}'
        check = _Checkpointer(file_format, False)
        try:
            check.save_checkpoint(state)
        except:
            return

        self.fail('No error was thrown when wrong format chosen for save file format')

    @patch("bink.Model.save")
    @patch('os.rename')
    def test_save_checkpoint_overwrite_recent(self, mock_save, mock_os_rename):
        state = {
            'self': Model(None, None, None, []),
            'epoch': 0,
            'metrics': {}
        }

        file_format = 'test_file_{epoch}'
        check = _Checkpointer(file_format, False)
        check.save_checkpoint(state, True)

        state['epoch'] = 1
        check.save_checkpoint(state)

        self.assertTrue(mock_os_rename.call_args_list[0][0][0] == 'test_file_0.pt')
        self.assertTrue(mock_os_rename.call_args_list[1][0][0] == 'test_file_1.pt')
        self.assertTrue(mock_os_rename.call_args_list[0][0][1] == 'test_file_0.bink')
        self.assertTrue(mock_os_rename.call_args_list[1][0][1] == 'test_file_1.bink')

class Test_MostRecent(TestCase):
    @patch('bink.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_save(self, mock_save_check):
        state = {}
        check = MostRecent('test_file')

        check.on_end_epoch(state)
        check.on_end_epoch(state)

        self.assertTrue(mock_save_check.call_count == 2)

class Test_Interval(TestCase):
    @patch('bink.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_interval_is_1(self, mock_save_check):
        state = {}
        check = Interval('test_file', period=1)

        check.on_end_epoch(state)
        check.on_end_epoch(state)

        self.assertTrue(mock_save_check.call_count == 2)

    @patch('bink.callbacks.checkpointers._Checkpointer.save_checkpoint')
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

    #TODO: Negative and fractional interval test and decide how to handle

class Test_Best(TestCase):
    @patch('bink.callbacks.checkpointers._Checkpointer.save_checkpoint')
    def test_min(self, mock_save):