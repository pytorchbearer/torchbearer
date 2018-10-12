from unittest import TestCase
from unittest.mock import MagicMock, Mock

import torchbearer
from torchbearer.callbacks import CallbackList, Callback, Tqdm, TensorBoard


class TestCallback(TestCase):
    def test_state_dict(self):
        callback = Callback()

        self.assertEqual(callback.state_dict(), {})
        self.assertEqual(callback.load_state_dict({}), callback)

    def test_str(self):
        callback = Callback()
        self.assertEqual(str(callback).strip(), "torchbearer.callbacks.callbacks.Callback")

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
        self.assertIsNone(callback.on_checkpoint({}))
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
        self.callback_1 = MagicMock(spec=torchbearer.callbacks.printer.Tqdm())
        self.callback_2 = MagicMock(spec=torchbearer.callbacks.tensor_board.TensorBoard())
        callbacks = [self.callback_1, self.callback_2]
        self.list = CallbackList(callbacks)

    def test_state_dict(self):
        self.callback_1.state_dict = Mock(return_value='test_1')
        self.callback_2.state_dict = Mock(return_value='test_2')

        state = self.list.state_dict()

        self.assertEqual(self.callback_1.state_dict.call_count, 1)
        self.assertEqual(self.callback_2.state_dict.call_count, 1)
        self.assertEqual(state[CallbackList.CALLBACK_STATES][0], 'test_1')
        self.assertEqual(state[CallbackList.CALLBACK_STATES][1], 'test_2')
        self.assertEqual(state[CallbackList.CALLBACK_TYPES][0], Tqdm().__class__)
        self.assertEqual(state[CallbackList.CALLBACK_TYPES][1], TensorBoard().__class__)

    def test_load_state_dict(self):
        self.callback_1.load_state_dict = Mock(return_value='test_1')
        self.callback_2.load_state_dict = Mock(return_value='test_2')

        self.callback_1.state_dict = Mock(return_value='test_1')
        self.callback_2.state_dict = Mock(return_value='test_2')

        state = self.list.state_dict()
        self.list.load_state_dict(state)

        self.callback_1.load_state_dict.assert_called_once_with('test_1')
        self.callback_2.load_state_dict.assert_called_once_with('test_2')

        state = self.list.state_dict()
        state[CallbackList.CALLBACK_TYPES] = list(reversed(state[CallbackList.CALLBACK_TYPES]))

        with self.assertWarns(UserWarning, msg='Callback classes did not match, expected: {\'TensorBoard\', \'Tqdm\'}'):
            self.list.load_state_dict(state)

    def test_for_list(self):
        self.list.on_start({})
        self.assertTrue(self.callback_1.method_calls[0][0] == 'on_start')
        self.assertTrue(self.callback_2.method_calls[0][0] == 'on_start')

    def test_list_in_list(self):
        callback = 'test'
        clist = CallbackList([callback])
        clist2 = CallbackList([clist])
        self.assertTrue(clist2.callback_list[0] == 'test')
