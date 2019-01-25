from unittest import TestCase
from mock import MagicMock, Mock

import torchbearer
from torchbearer.callbacks import CallbackList, Tqdm, TensorBoard


class TestCallbackList(TestCase):
    def __init__(self, methodName='runTest'):
        super(TestCallbackList, self).__init__(methodName)
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

        import warnings
        with warnings.catch_warnings(record=True) as w:
            self.list.load_state_dict(state)
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue('Callback classes did not match, expected: [\'TensorBoard\', \'Tqdm\']' in str(w[-1].message))

    def test_for_list(self):
        self.list.on_start({})
        self.assertTrue(self.callback_1.method_calls[0][0] == 'on_start')
        self.assertTrue(self.callback_2.method_calls[0][0] == 'on_start')

    def test_list_in_list(self):
        callback = 'test'
        clist = CallbackList([callback])
        clist2 = CallbackList([clist])
        self.assertTrue(clist2.callback_list[0] == 'test')

    def test_iter_copy(self):
        callback = 'test'
        clist = CallbackList([callback])
        cpy = clist.__copy__()
        self.assertTrue(cpy.callback_list[0] == 'test')
        self.assertTrue(cpy is not clist)
        cpy = clist.copy()
        self.assertTrue(cpy.callback_list[0] == 'test')
        self.assertTrue(cpy is not clist)
        for cback in clist:
            self.assertTrue(cback == 'test')
