import sys
from unittest import TestCase

from mock import MagicMock, patch, ANY

import torchbearer
from torchbearer.callbacks import PyCM

import matplotlib.pyplot as plt  # Import so that it can be mocked
plt.ioff()


class TestHandlers(TestCase):
    @patch('matplotlib.pyplot')
    def test_to_pyplot(self, mock_pyplot):
        if sys.version_info[0] >= 3:
            import pycm

            handler = torchbearer.callbacks.pycm._to_pyplot(True, 'test {epoch}')

            y_true = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
            y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
            cm = pycm.ConfusionMatrix(y_true, y_pred)
            handler(cm, {torchbearer.EPOCH: 3})

            self.assertTrue(mock_pyplot.imshow.call_args[0][0].max() == 1)
            mock_pyplot.title.assert_called_once_with('test 3')

            handler = torchbearer.callbacks.pycm._to_pyplot(False)

            y_true = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
            y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
            cm = pycm.ConfusionMatrix(y_true, y_pred)
            handler(cm, {})

            self.assertTrue(mock_pyplot.imshow.call_args[0][0].max() > 1)


class TestPyCM(TestCase):
    def test_exception(self):
        if sys.version_info[0] < 3:
            self.assertRaises(Exception, PyCM)

    @patch('torchbearer.callbacks.pycm.EpochLambda')
    def test_make_cm(self, emock_lambda):
        if sys.version_info[0] >= 3:
            with patch('pycm.ConfusionMatrix') as confusion_mocktrix:
                confusion_mocktrix.return_value = 'test'
                handler = MagicMock()
                callback = PyCM(test=10).with_handler(handler)
                state = {torchbearer.METRIC_LIST: None}

                callback._add_metric(state)
                emock_lambda.assert_called_once_with('pycm', ANY, False)

                make_cm = emock_lambda.call_args[0][1]

                import torch

                y_pred = torch.rand(5, 2) / 2
                y_pred[:, 1] = 1
                y_true = MagicMock()

                make_cm(y_pred, y_true)

                self.assertTrue(y_true.cpu.call_count == 1)
                self.assertTrue(y_true.cpu().numpy.call_count == 1)
                confusion_mocktrix.assert_called_once_with(y_true.cpu().numpy(), ANY, test=10)
                self.assertTrue(confusion_mocktrix.call_args[0][1].sum() == 5)
                handler.assert_called_once_with('test', state)

    def test_on_train(self):
        if sys.version_info[0] >= 3:
            callback = PyCM().on_train()
            state = {torchbearer.METRIC_LIST: None}
            callback.on_start_training(state)
            self.assertTrue(state[torchbearer.METRIC_LIST] is not None)

    def test_on_val(self):
        if sys.version_info[0] >= 3:
            callback = PyCM().on_val()
            state = {torchbearer.METRIC_LIST: None, torchbearer.DATA: torchbearer.VALIDATION_DATA}
            callback.on_start_validation(state)
            self.assertTrue(state[torchbearer.METRIC_LIST] is not None)

    def test_on_test(self):
        if sys.version_info[0] >= 3:
            callback = PyCM().on_test()
            state = {torchbearer.METRIC_LIST: None, torchbearer.DATA: torchbearer.TEST_DATA}
            callback.on_start_validation(state)
            self.assertTrue(state[torchbearer.METRIC_LIST] is not None)

    def test_with_handler(self):
        if sys.version_info[0] >= 3:
            callback = PyCM()
            callback.with_handler('test')
            self.assertTrue('test' in callback._handlers)

    def test_to_state(self):
        if sys.version_info[0] >= 3:
            callback = PyCM()
            callback.to_state('test')
            out = {}
            callback._handlers[0]('cm', out)
            self.assertTrue('test' in out)
            self.assertTrue(out['test'] == 'cm')

    @patch('torchbearer.callbacks.pycm.print')
    def test_to_console(self, mock_print):
        if sys.version_info[0] >= 3:
            callback = PyCM()
            callback.to_console()
            callback._handlers[0]('cm', {})
            mock_print.assert_called_once_with('cm')

    def test_to_file(self):
        if sys.version_info[0] >= 3:
            callback = PyCM()
            callback.to_pycm_file('test {epoch}')
            cm = MagicMock()
            callback._handlers[0](cm, {torchbearer.EPOCH: 1})

            cm.save_stat.assert_called_once_with('test 1', address=True, overall_param=None, class_param=None,
                                                 class_name=None)

            callback = PyCM()
            callback.to_html_file('test {epoch}')
            cm = MagicMock()
            callback._handlers[0](cm, {torchbearer.EPOCH: 2})

            cm.save_html.assert_called_once_with('test 2', address=True, overall_param=None, class_param=None,
                                                 class_name=None, color=(0, 0, 0), normalize=False)

            callback = PyCM()
            callback.to_csv_file('test {epoch}')
            cm = MagicMock()
            callback._handlers[0](cm, {torchbearer.EPOCH: 3})

            cm.save_csv.assert_called_once_with('test 3', address=True, overall_param=None, class_param=None,
                                                class_name=None, matrix_save=True, normalize=False)

            callback = PyCM()
            callback.to_obj_file('test {epoch}')
            cm = MagicMock()
            callback._handlers[0](cm, {torchbearer.EPOCH: 4})

            cm.save_obj.assert_called_once_with('test 4', address=True, save_stat=False, save_vector=True)

    @patch('torchbearer.callbacks.pycm._to_pyplot')
    def test_to_pyplot(self, mock_to_pyplot):
        if sys.version_info[0] >= 3:
            PyCM().to_pyplot(True, 'test', 'test2')
            mock_to_pyplot.assert_called_once_with(normalize=True, title='test', cmap='test2')