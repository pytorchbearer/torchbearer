import unittest
from mock import Mock, MagicMock, create_autospec

import torch

import torchbearer
from torchbearer.bases import Metric, Callback, base_closure
from torchbearer.metrics.primitives import CategoricalAccuracy


class TestMetric(unittest.TestCase):
    def setUp(self):
        self._state = {
            torchbearer.Y_TRUE: torch.LongTensor([0, 1, 2, 2, 1]),
            torchbearer.Y_PRED: torch.FloatTensor([
                [0.9, 0.1, 0.1], # Correct
                [0.1, 0.9, 0.1], # Correct
                [0.1, 0.1, 0.9], # Correct
                [0.9, 0.1, 0.1], # Incorrect
                [0.9, 0.1, 0.1], # Incorrect
            ])
        }
        self._state[torchbearer.Y_PRED].requires_grad = True
        self._targets = [1, 1, 1, 0, 0]
        self._metric = CategoricalAccuracy().root

    def test_requires_grad(self):
        result = self._metric.process(self._state)
        self.assertTrue(self._state[torchbearer.Y_PRED].requires_grad is True)
        self.assertTrue(result.requires_grad is False)

    def test_empty_methods(self):
        metric = Metric(name='test')
        self.assertTrue(metric.process() is None)
        self.assertTrue(metric.process_final() is None)


class TestCallback(unittest.TestCase):
    def test_state_dict(self):
        callback = Callback()

        self.assertEqual(callback.state_dict(), {})
        self.assertEqual(callback.load_state_dict({}), callback)

    def test_str(self):
        callback = Callback()
        self.assertEqual(str(callback).strip(), "torchbearer.bases.Callback")

    def test_empty_methods(self):
        callback = Callback()

        self.assertIsNone(callback.on_init({}))
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


class TestBaseCrit(unittest.TestCase):
    def test_opt(self):
        opt = Mock()
        opt.zero_grad = Mock()
        state = {torchbearer.X: None, torchbearer.MODEL: lambda x: None, torchbearer.Y_TRUE: None,
                 torchbearer.CRITERION: lambda x,y: MagicMock(), torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: MagicMock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)
        closure(state)
        self.assertTrue(opt.zero_grad.call_count == 1)

    def test_forward_x(self):
        opt = Mock()

        def model_forward(x):
            return None
        model = create_autospec(model_forward)

        x = 'test'

        state = {torchbearer.X: x, torchbearer.MODEL: model, torchbearer.Y_TRUE: None,
                 torchbearer.CRITERION: lambda x,y: MagicMock(), torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: MagicMock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)
        closure(state)
        self.assertTrue(model.call_args[0][0] == x)

    def test_forward_state(self):
        opt = Mock()

        def model_forward(x, state):
            return None
        model = create_autospec(model_forward)

        x = 'test'

        state = {torchbearer.X: x, torchbearer.MODEL: model, torchbearer.Y_TRUE: None,
                 torchbearer.CRITERION: lambda x,y: MagicMock(), torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: MagicMock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)
        closure(state)
        self.assertTrue(model.call_args[0][0] == x)
        self.assertDictEqual(model.call_args[1]['state'], state)

    def test_loss_no_state(self):
        opt = Mock()

        y_pred = 'yp'
        y_true = 'yt'
        def loss_sig(y_pred, y_true):
            return None
        crit = create_autospec(loss_sig)

        state = {torchbearer.X: None, torchbearer.MODEL: lambda x: y_pred, torchbearer.Y_TRUE: y_true,
                 torchbearer.CRITERION: crit, torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: MagicMock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)
        closure(state)
        self.assertTrue(crit.call_args[0] == (y_pred, y_true))

    def test_loss_state(self):
        opt = Mock()

        y_pred = 'yp'
        y_true = 'yt'
        def loss_sig(state):
            return None
        crit = create_autospec(loss_sig)

        state = {torchbearer.X: None, torchbearer.MODEL: lambda x: y_pred, torchbearer.Y_TRUE: y_true,
                 torchbearer.CRITERION: crit, torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: MagicMock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)
        closure(state)
        self.assertDictEqual(crit.call_args[0][0], state)

    def test_backward(self):
        opt = Mock()
        loss = Mock()
        loss.backward = Mock()

        state = {torchbearer.X: None, torchbearer.MODEL: lambda x: None, torchbearer.Y_TRUE: None,
                 torchbearer.CRITERION: lambda state: loss, torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: MagicMock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)
        closure(state)
        self.assertTrue(loss.backward.call_count == 1)

    def test_callback_list(self):
        opt = Mock()
        callback_list = Mock()
        callback_list.on_forward = Mock()
        callback_list.on_criterion = Mock()
        callback_list.on_backward = Mock()

        state = {torchbearer.X: None, torchbearer.MODEL: lambda x: None, torchbearer.Y_TRUE: None,
                 torchbearer.CRITERION: lambda state: MagicMock(), torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: callback_list, torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)
        closure(state)
        self.assertTrue(callback_list.on_forward.call_count == 1)
        self.assertTrue(callback_list.on_criterion.call_count == 1)
        self.assertTrue(callback_list.on_backward.call_count == 1)
