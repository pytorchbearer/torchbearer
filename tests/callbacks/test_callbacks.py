from unittest import TestCase
from unittest.mock import MagicMock

from torchbearer.callbacks import CallbackList


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
