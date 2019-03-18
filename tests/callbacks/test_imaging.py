from unittest import TestCase

from mock import MagicMock, patch

import torchbearer
import torchbearer.callbacks.imaging as imaging


class TestImagingCallback(TestCase):
    def test_transform(self):
        callback = imaging.ImagingCallback()
        self.assertTrue(callback.transform('test') is 'test')

        callback = imaging.ImagingCallback(transform=lambda _: 'test')
        self.assertTrue(callback.transform('something else') is 'test')

    def test_process(self):
        callback = imaging.ImagingCallback()
        callback.on_batch = lambda _: 'test'
        handler = MagicMock()
        callback = callback.with_handler(handler)
        callback.transform = MagicMock(return_value='test')
        state = 'state'
        callback.process(state)
        handler.assert_called_once_with('test', 'state')
        self.assertTrue(callback.transform.call_count == 1)

    def test_simple_methods(self):
        callback = imaging.ImagingCallback()
        self.assertRaises(NotImplementedError, lambda : callback.on_batch('test'))

        callback = imaging.ImagingCallback().to_file('test')
        self.assertTrue('_to_file' in str(callback._handlers[0]))

        callback = imaging.ImagingCallback().to_pyplot()
        self.assertTrue('_to_pyplot' in str(callback._handlers[0]))

        callback = imaging.ImagingCallback().to_state('test')
        state = {}
        callback._handlers[0]('image', state)
        self.assertTrue('test' in state)
        self.assertTrue(state['test'] is 'image')

        callback = imaging.ImagingCallback().to_tensorboard()
        self.assertTrue('_to_tensorboard' in str(callback._handlers[0]))

        callback = imaging.ImagingCallback().to_visdom()
        self.assertTrue('_to_visdom' in str(callback._handlers[0]))

    def test_on_train(self):
        callback = imaging.ImagingCallback()
        mock = MagicMock()
        callback.on_step_training = mock
        callback.process = MagicMock()
        callback = callback.on_train()
        callback.on_step_training('state')
        mock.assert_called_once_with('state')
        callback.process.assert_called_once_with('state')

    def test_on_val(self):
        callback = imaging.ImagingCallback()
        mock = MagicMock()
        callback.on_step_validation = mock
        callback.process = MagicMock()
        callback = callback.on_val()
        state = {torchbearer.DATA: torchbearer.TEST_DATA}
        callback.on_step_validation(state)
        mock.assert_called_once_with(state)
        self.assertTrue(callback.process.call_count == 0)

        mock.reset_mock()
        callback.process.reset_mock()

        state = {torchbearer.DATA: torchbearer.VALIDATION_DATA}
        callback.on_step_validation(state)
        mock.assert_called_once_with(state)
        callback.process.assert_called_once_with(state)

    def test_on_test(self):
        callback = imaging.ImagingCallback()
        mock = MagicMock()
        callback.on_step_validation = mock
        callback.process = MagicMock()
        callback = callback.on_test()
        state = {torchbearer.DATA: torchbearer.VALIDATION_DATA}
        callback.on_step_validation(state)
        mock.assert_called_once_with(state)
        self.assertTrue(callback.process.call_count == 0)

        mock.reset_mock()
        callback.process.reset_mock()

        state = {torchbearer.DATA: torchbearer.TEST_DATA}
        callback.on_step_validation(state)
        mock.assert_called_once_with(state)
        callback.process.assert_called_once_with(state)
