import unittest
from mock import Mock, MagicMock, create_autospec, patch

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

    def state_model_with_e(self, e):
        def model(x, state):
            raise e
            return x
        return model

    def stateless_model_with_e(self, e):
        def model(x):
            raise e
            return x
        return model

    def optional_model_with_e(self, e):
        def model(x, state=None):
            raise e
            return x
        return model

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

    def test_forward_multiple_x(self):
        opt = Mock()

        def model_forward(x1, x2):
            return None
        model = create_autospec(model_forward)

        x1 = 'test1'
        x2 = 'test2'

        state = {torchbearer.X: [x1, x2], torchbearer.MODEL: model, torchbearer.Y_TRUE: None,
                 torchbearer.CRITERION: lambda x,y: MagicMock(), torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: MagicMock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)
        closure(state)
        self.assertTrue(model.call_args[0][0] == x1)
        self.assertTrue(model.call_args[0][1] == x2)

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

    def test_forward_multiple_x_and_state(self):
        opt = Mock()

        def model_forward(x1, x2, state):
            return None
        model = create_autospec(model_forward)

        x1 = 'test1'
        x2 = 'test2'

        state = {torchbearer.X: [x1, x2], torchbearer.MODEL: model, torchbearer.Y_TRUE: None,
                 torchbearer.CRITERION: lambda x,y: MagicMock(), torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: MagicMock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)
        closure(state)
        self.assertTrue(model.call_args[0][0] == x1)
        self.assertTrue(model.call_args[0][1] == x2)
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

    def test_loss_multiple_output_no_state(self):
        opt = Mock()

        y_pred1 = 'yp1'
        y_pred2 = 'yp2'
        y_true1 = 'yt1'
        y_true2 = 'yt2'
        def loss_sig(y_pred1, y_pred2, y_true1, y_true2):
            return None
        crit = create_autospec(loss_sig)

        state = {torchbearer.X: None, torchbearer.MODEL: lambda x: (y_pred1, y_pred2), torchbearer.Y_TRUE: (y_true1, y_true2),
                 torchbearer.CRITERION: crit, torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: MagicMock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)
        closure(state)
        self.assertTrue(crit.call_args[0] == (y_pred1, y_pred2, y_true1, y_true2))

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


    def test_stateless_type_error(self):
        state = {torchbearer.X: None, torchbearer.MODEL: self.stateless_model_with_e(TypeError('test')), torchbearer.CRITERION: lambda state: MagicMock(),
                 torchbearer.OPTIMIZER: Mock(), torchbearer.CALLBACK_LIST: Mock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)

        with self.assertRaises(Exception) as context:
            closure(state)

        self.assertTrue(len(context.exception.args[0]) == 2)

    def test_stateless_exception(self):
        state = {torchbearer.X: None, torchbearer.MODEL: self.stateless_model_with_e(Exception('test')), torchbearer.CRITERION: lambda state: MagicMock(),
                 torchbearer.OPTIMIZER: Mock(), torchbearer.CALLBACK_LIST: Mock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)

        with self.assertRaises(Exception) as context:
            closure(state)

        self.assertTrue(len(context.exception.args[0]) == 1)
        self.assertTrue('test' in context.exception.args[0][0].args)

    def test_exception_exception(self):

        state = {torchbearer.X: None, torchbearer.MODEL: self.optional_model_with_e(Exception('test')),
                 torchbearer.CRITERION: lambda state: MagicMock(),
                 torchbearer.OPTIMIZER: Mock(), torchbearer.CALLBACK_LIST: Mock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)

        with self.assertRaises(Exception) as context:
            closure(state)

        self.assertTrue(len(context.exception.args[0]) == 2)
        self.assertTrue('test' in context.exception.args[0][0].args)
        self.assertTrue('test' in context.exception.args[0][1].args)


    def test_state_exception(self):
        state = {torchbearer.X: None, torchbearer.MODEL: self.state_model_with_e(Exception('test')), torchbearer.CRITERION: lambda state: MagicMock(),
                 torchbearer.OPTIMIZER: Mock(), torchbearer.CALLBACK_LIST: Mock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)

        with self.assertRaises(Exception) as context:
            closure(state)

        self.assertTrue(len(context.exception.args[0]) == 1)
        self.assertTrue('test' in context.exception.args[0][0].args)

    def test_state_type_error(self):
        state = {torchbearer.X: None, torchbearer.MODEL: self.state_model_with_e(TypeError('test')), torchbearer.CRITERION: lambda state: MagicMock(),
                 torchbearer.OPTIMIZER: Mock(), torchbearer.CALLBACK_LIST: Mock(), torchbearer.BACKWARD_ARGS: {}}

        closure = base_closure(torchbearer.X, torchbearer.MODEL, torchbearer.Y_PRED, torchbearer.Y_TRUE,
                               torchbearer.CRITERION, torchbearer.LOSS, torchbearer.OPTIMIZER)

        with self.assertRaises(Exception) as context:
            closure(state)

        self.assertTrue(len(context.exception.args[0]) == 2)


class TestApexCrit(unittest.TestCase):
    def setUp(self):
        super(TestApexCrit, self).setUp()

        import types
        import sys

        module_name = 'apex'
        bogus_module = types.ModuleType(module_name)
        self.old_module = sys.modules[module_name] if module_name in sys.modules else None
        sys.modules[module_name] = bogus_module
        self.mock_amp = MagicMock(name=module_name + '.amp')
        bogus_module.amp = self.mock_amp

        from torchbearer.bases import apex_closure
        self.closure = apex_closure()

    def tearDown(self):
        super(TestApexCrit, self).tearDown()
        if self.old_module is not None:
            import sys
            sys.modules['apex'] = self.old_module

    def state_model_with_e(self, e):
        def model(x, state):
            raise e
            return x
        return model

    def stateless_model_with_e(self, e):
        def model(x):
            raise e
            return x
        return model

    def test_opt(self):
        opt = Mock()
        opt.zero_grad = Mock()
        state = {torchbearer.X: None, torchbearer.MODEL: lambda x: None, torchbearer.Y_TRUE: None,
                 torchbearer.CRITERION: lambda x,y: MagicMock(), torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: MagicMock(), torchbearer.BACKWARD_ARGS: {}}

        closure = self.closure
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

        closure = self.closure
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

        closure = self.closure
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

        closure = self.closure
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

        closure = self.closure
        closure(state)
        self.assertDictEqual(crit.call_args[0][0], state)

    def test_backward(self):
        opt = Mock()
        loss = Mock()
        loss.backward = Mock()

        state = {torchbearer.X: None, torchbearer.MODEL: lambda x: None, torchbearer.Y_TRUE: None,
                 torchbearer.CRITERION: lambda state: loss, torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: MagicMock(), torchbearer.BACKWARD_ARGS: {}}

        closure = self.closure
        closure(state)
        self.assertTrue('backward' in self.mock_amp.scale_loss.mock_calls[2][0])

    def test_callback_list(self):
        opt = Mock()
        callback_list = Mock()
        callback_list.on_forward = Mock()
        callback_list.on_criterion = Mock()
        callback_list.on_backward = Mock()

        state = {torchbearer.X: None, torchbearer.MODEL: lambda x: None, torchbearer.Y_TRUE: None,
                 torchbearer.CRITERION: lambda state: MagicMock(), torchbearer.LOSS: None, torchbearer.OPTIMIZER: opt,
                 torchbearer.CALLBACK_LIST: callback_list, torchbearer.BACKWARD_ARGS: {}}

        closure = self.closure
        closure(state)
        self.assertTrue(callback_list.on_forward.call_count == 1)
        self.assertTrue(callback_list.on_criterion.call_count == 1)
        self.assertTrue(callback_list.on_backward.call_count == 1)


    def test_stateless_type_error(self):
        state = {torchbearer.X: None, torchbearer.MODEL: self.stateless_model_with_e(TypeError('test')), torchbearer.CRITERION: lambda state: MagicMock(),
                 torchbearer.OPTIMIZER: Mock(), torchbearer.CALLBACK_LIST: Mock(), torchbearer.BACKWARD_ARGS: {}}

        closure = self.closure

        with self.assertRaises(Exception) as context:
            closure(state)

        self.assertTrue(len(context.exception.args[0]) == 2)

    def test_stateless_exception(self):
        state = {torchbearer.X: None, torchbearer.MODEL: self.stateless_model_with_e(Exception('test')), torchbearer.CRITERION: lambda state: MagicMock(),
                 torchbearer.OPTIMIZER: Mock(), torchbearer.CALLBACK_LIST: Mock(), torchbearer.BACKWARD_ARGS: {}}

        closure = self.closure

        with self.assertRaises(Exception) as context:
            closure(state)

        self.assertTrue(len(context.exception.args[0]) == 1)
        self.assertTrue('test' in context.exception.args[0][0].args)

    def test_state_exception(self):
        state = {torchbearer.X: None, torchbearer.MODEL: self.state_model_with_e(Exception('test')), torchbearer.CRITERION: lambda state: MagicMock(),
                 torchbearer.OPTIMIZER: Mock(), torchbearer.CALLBACK_LIST: Mock(), torchbearer.BACKWARD_ARGS: {}}

        closure = self.closure

        with self.assertRaises(Exception) as context:
            closure(state)

        self.assertTrue(len(context.exception.args[0]) == 1)
        self.assertTrue('test' in context.exception.args[0][0].args)

    def test_state_type_error(self):
        state = {torchbearer.X: None, torchbearer.MODEL: self.state_model_with_e(TypeError('test')), torchbearer.CRITERION: lambda state: MagicMock(),
                 torchbearer.OPTIMIZER: Mock(), torchbearer.CALLBACK_LIST: Mock(), torchbearer.BACKWARD_ARGS: {}}

        closure = self.closure

        with self.assertRaises(Exception) as context:
            closure(state)

        self.assertTrue(len(context.exception.args[0]) == 2)