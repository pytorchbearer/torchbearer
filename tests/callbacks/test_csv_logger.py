from unittest import TestCase
from mock import patch, mock_open

import torchbearer
from torchbearer.callbacks import CSVLogger


class TestCSVLogger(TestCase):

    @patch("torchbearer.callbacks.csv_logger.open", new_callable=mock_open)
    def test_write_header(self, mock_open):
        state = {
            torchbearer.EPOCH: 0,
            torchbearer.BATCH: 1,
            torchbearer.METRICS: {'test_metric_1': 0.1, 'test_metric_2': 5}
        }

        logger = CSVLogger('test_file.log')

        logger.on_step_training(state)
        logger.on_end_epoch(state)
        logger.on_end(state)

        header = mock_open.mock_calls[1][1][0]
        self.assertTrue('epoch' in header)
        self.assertTrue('test_metric_1' in header)
        self.assertTrue('test_metric_2' in header)

    @patch("torchbearer.callbacks.csv_logger.open", new_callable=mock_open)
    def test_write_no_header(self, mock_open):
        state = {
            torchbearer.EPOCH: 0,
            torchbearer.BATCH: 1,
            torchbearer.METRICS: {'test_metric_1': 0.1, 'test_metric_2': 5}
        }

        logger = CSVLogger('test_file.log', write_header=False)

        logger.on_step_training(state)
        logger.on_end_epoch(state)
        logger.on_end(state)

        header = mock_open.mock_calls[1][1][0]
        self.assertTrue('epoch' not in header)
        self.assertTrue('test_metric_1' not in header)
        self.assertTrue('test_metric_2' not in header)

    @patch("torchbearer.callbacks.csv_logger.open", new_callable=mock_open)
    def test_csv_closed(self, mock_open):
        state = {
            torchbearer.EPOCH: 0,
            torchbearer.BATCH: 1,
            torchbearer.METRICS: {'test_metric_1': 0.1, 'test_metric_2': 5}
        }

        logger = CSVLogger('test_file.log', write_header=False)

        logger.on_step_training(state)
        logger.on_end_epoch(state)
        logger.on_end(state)

        self.assertTrue(mock_open.return_value.close.called)

    @patch("torchbearer.callbacks.csv_logger.open", new_callable=mock_open)
    def test_append(self, mock_open):
        state = {
            torchbearer.EPOCH: 0,
            torchbearer.BATCH: 1,
            torchbearer.METRICS: {'test_metric_1': 0.1, 'test_metric_2': 5}
        }

        logger = CSVLogger('test_file.log', append=True)

        logger.on_step_training(state)
        logger.on_end_epoch(state)
        logger.on_end(state)

        import sys
        if sys.version_info[0] < 3:
            self.assertTrue(mock_open.call_args[0][1] == 'ab')
        else:
            self.assertTrue(mock_open.call_args[0][1] == 'a')

    @patch("torchbearer.callbacks.csv_logger.open", new_callable=mock_open)
    def test_get_field_dict(self, mock_open):
        state = {
            torchbearer.EPOCH: 0,
            torchbearer.BATCH: 1,
            torchbearer.METRICS: {'test_metric_1': 0.1, 'test_metric_2': 5}
        }
        correct_fields_dict = {
            'epoch': 0,
            'batch': 1,
            'test_metric_1': 0.1,
            'test_metric_2': 5
        }

        logger = CSVLogger('test_file.log', batch_granularity=True)

        logger_fields_dict = logger._get_field_dict(state)

        self.assertDictEqual(logger_fields_dict, correct_fields_dict)

    @patch('torchbearer.callbacks.CSVLogger._write_to_dict')
    @patch("torchbearer.callbacks.csv_logger.open", new_callable=mock_open)
    def test_write_on_epoch(self, mock_open, mock_write):
        state = {
            torchbearer.EPOCH: 0,
            torchbearer.BATCH: 1,
            torchbearer.METRICS: {'test_metric_1': 0.1, 'test_metric_2': 5}
        }

        logger = CSVLogger('test_file.log')

        logger.on_step_training(state)
        logger.on_end_epoch(state)
        logger.on_end(state)

        self.assertEqual(mock_write.call_count, 1)

    @patch('torchbearer.callbacks.CSVLogger._write_to_dict')
    @patch("torchbearer.callbacks.csv_logger.open", new_callable=mock_open)
    def test_batch_granularity(self, mock_open, mock_write):
        state = {
            torchbearer.EPOCH: 0,
            torchbearer.BATCH: 1,
            torchbearer.METRICS: {'test_metric_1': 0.1, 'test_metric_2': 5}
        }

        logger = CSVLogger('test_file.log', batch_granularity=True)

        logger.on_step_training(state)
        logger.on_step_training(state)
        logger.on_end_epoch(state)
        logger.on_end(state)

        self.assertTrue(mock_write.call_count == 3)
