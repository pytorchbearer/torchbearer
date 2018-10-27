from unittest import TestCase
from unittest.mock import patch, Mock, MagicMock

import torchbearer
from torchbearer.callbacks import Tqdm, ConsolePrinter


class TestFormatMetrics(TestCase):
    def test_precision(self):
        metrics = {'test': 1.2345}
        res = torchbearer.callbacks.printer._format_metrics(metrics, 3)
        self.assertEqual(res, 'test=1.23')

    def test_string(self):
        metrics = {'test': '1.2345'}
        res = torchbearer.callbacks.printer._format_metrics(metrics, 3)
        self.assertEqual(res, 'test=1.2345')

    def test_not_string(self):
        metrics = {'test': {'hello': 2}}
        res = torchbearer.callbacks.printer._format_metrics(metrics, 3)
        self.assertEqual(res, 'test={\'hello\': 2}')


class TestConsolePrinter(TestCase):
    @patch('builtins.print')
    def test_console_printer(self, mock_print):
        state = {torchbearer.BATCH: 5, torchbearer.EPOCH: 1, torchbearer.MAX_EPOCHS: 10, torchbearer.TRAIN_STEPS: 100, torchbearer.VALIDATION_STEPS: 101, torchbearer.METRICS: {'test': 0.99456}}
        printer = ConsolePrinter(validation_label_letter='e')
        state[torchbearer.STEPS] = state[torchbearer.TRAIN_STEPS]

        printer.on_step_training(state)
        mock_print.assert_called_once_with('\r1/10(t): 5/100 test=0.9946', end='')
        mock_print.reset_mock()

        printer.on_end_training(state)
        mock_print.assert_called_once_with('\r1/10(t): test=0.9946')
        mock_print.reset_mock()

        state[torchbearer.STEPS] = state[torchbearer.VALIDATION_STEPS]
        printer.on_step_validation(state)
        mock_print.assert_called_once_with('\r1/10(e): 5/101 test=0.9946', end='')
        mock_print.reset_mock()

        printer.on_end_validation(state)
        mock_print.assert_called_once_with('\r1/10(e): test=0.9946')
        mock_print.reset_mock()


class TestTqdm(TestCase):
    def test_tqdm(self):
        state = {torchbearer.EPOCH: 1, torchbearer.MAX_EPOCHS: 10, torchbearer.TRAIN_STEPS: 100, torchbearer.VALIDATION_STEPS: 101, torchbearer.METRICS: {'test': 10}}
        tqdm = Tqdm(validation_label_letter='e')
        tqdm.tqdm_module = MagicMock()
        mock_tqdm = tqdm.tqdm_module
        state[torchbearer.STEPS] = state[torchbearer.TRAIN_STEPS]

        tqdm.on_start_training(state)
        mock_tqdm.assert_called_once_with(total=100, desc='1/10(t)')

        tqdm.on_step_training(state)
        mock_tqdm.return_value.set_postfix_str.assert_called_once_with('test=10')
        mock_tqdm.return_value.update.assert_called_once_with(1)
        mock_tqdm.return_value.set_postfix_str.reset_mock()

        tqdm.on_end_training(state)
        mock_tqdm.return_value.set_postfix_str.assert_called_once_with('test=10')
        self.assertEqual(mock_tqdm.return_value.close.call_count, 1)

        mock_tqdm.reset_mock()
        mock_tqdm.return_value.set_postfix_str.reset_mock()
        mock_tqdm.return_value.update.reset_mock()
        mock_tqdm.return_value.close.reset_mock()

        state[torchbearer.STEPS] = state[torchbearer.VALIDATION_STEPS]
        tqdm.on_start_validation(state)
        mock_tqdm.assert_called_once_with(total=101, desc='1/10(e)')

        tqdm.on_step_validation(state)
        mock_tqdm.return_value.set_postfix_str.assert_called_once_with('test=10')
        mock_tqdm.return_value.update.assert_called_once_with(1)
        mock_tqdm.return_value.set_postfix_str.reset_mock()

        tqdm.on_end_validation(state)
        mock_tqdm.return_value.set_postfix_str.assert_called_once_with('test=10')
        self.assertEqual(mock_tqdm.return_value.close.call_count, 1)

    def test_tqdm_custom_args(self):
        state = {torchbearer.EPOCH: 1, torchbearer.MAX_EPOCHS: 10, torchbearer.TRAIN_STEPS: 100,
                 torchbearer.VALIDATION_STEPS: 101, torchbearer.METRICS: {'test': 10}}
        state[torchbearer.HISTORY] = [[[], state[torchbearer.METRICS]]]
        tqdm = Tqdm(ascii=True)
        state[torchbearer.STEPS] = state[torchbearer.TRAIN_STEPS]

        tqdm.tqdm_module = MagicMock()
        mock_tqdm = tqdm.tqdm_module

        tqdm.on_start_training(state)
        mock_tqdm.assert_called_once_with(total=100, desc='1/10(t)', ascii=True)

        tqdm = Tqdm(on_epoch=True, ascii=True)

        tqdm.tqdm_module = MagicMock()
        mock_tqdm = tqdm.tqdm_module

        tqdm.on_start(state)
        mock_tqdm.assert_called_once_with(initial=1, total=10, ascii=True)
