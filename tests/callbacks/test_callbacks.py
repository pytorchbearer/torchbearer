from unittest import TestCase
from unittest.mock import MagicMock

from torchbearer.callbacks import CallbackList, Callback


class TestCallback(TestCase):
    def test_empty_methods(self):
        callback = Callback()

        self.assertIsNone(callback.on_start({}))
        self.assertIsNone(callback.on_start_epoch({}))
        self.assertIsNone(callback.on_start_training({}))
        self.assertIsNone(callback.on_sample({}))
        self.assertIsNone(callback.on_forward({}))
        self.assertIsNone(callback.on_criterion({}))
        self.assertIsNone(callback.on_backward({}))
        self.assertIsNone(callback.on_step_training({}))
        self.assertIsNone(callback.on_end_training({}))
        self.assertIsNone(callback.on_end_epoch({}))
        self.assertIsNone(callback.on_end({}))
        self.assertIsNone(callback.on_start_validation({}))
        self.assertIsNone(callback.on_sample_validation({}))
        self.assertIsNone(callback.on_forward_validation({}))
        self.assertIsNone(callback.on_end_validation({}))
        self.assertIsNone(callback.on_step_validation({}))
        self.assertIsNone(callback.on_criterion_validation({}))


class TestCallbackList(TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.callback_1 = MagicMock()
        self.callback_2 = MagicMock()
        callbacks = [self.callback_1, self.callback_2]
        self.list = CallbackList(callbacks)

    def test_for_list(self):
        self.list.on_start({})
        self.assertTrue(self.callback_1.method_calls[0][0] == 'on_start')
        self.assertTrue(self.callback_2.method_calls[0][0] == 'on_start')

    def test_list_in_list(self):
        callback = 'test'
        clist = CallbackList([callback])
        clist2 = CallbackList([clist])
        self.assertTrue(clist2.callback_list[0] == 'test')
