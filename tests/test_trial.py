from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch, ANY

import torch
from torch.utils.data import DataLoader

import torchbearer as tb
import torchbearer.callbacks as callbacks
from torchbearer import Trial
from torchbearer.metrics import Metric
from torchbearer.trial import deep_to, load_batch_none, load_batch_predict, load_batch_standard, update_device_and_dtype, CallbackListInjection


class TestMockOptimizer(TestCase):
    @patch('torchbearer.trial.Optimizer')
    def test_mock_optimizer(self, mock_opt):
        mock_opt.add_param_group = Mock()
        mock_opt.load_state_dict = Mock()
        mock_opt.state_dict = Mock()
        mock_opt.step = Mock()
        mock_opt.zero_grad = Mock()

        opt = tb.trial.MockOptimizer()

        self.assertIsNone(opt.add_param_group({}))
        mock_opt.add_param_group.assert_not_called()

        self.assertIsNone(opt.load_state_dict({}))
        mock_opt.load_state_dict.assert_not_called()

        self.assertDictEqual(opt.state_dict(), {})
        mock_opt.state_dict.assert_not_called()

        self.assertIsNone(opt.step())
        mock_opt.step.assert_not_called()

        self.assertIsNone(opt.zero_grad())
        mock_opt.zero_grad.assert_not_called()


class TestCallbackListInjection(TestCase):
    def test_pass_through(self):
        mock = MagicMock()
        injection = CallbackListInjection(None, mock)

        # state_dict
        mock.state_dict.return_value = 'test'
        self.assertEqual(injection.state_dict(), 'test')
        self.assertEqual(mock.state_dict.call_count, 1)

        # load_state_dict
        injection.load_state_dict('test')
        mock.load_state_dict.assert_called_once_with('test')

        # iter
        mock.__iter__.return_value = ['iterator']
        self.assertEqual(next(injection.__iter__()), 'iterator')
        self.assertEqual(mock.__iter__.call_count, 1)

        # copy
        mock.copy.return_value = 'copy'
        self.assertEqual(injection.copy(), 'copy')

        # append
        injection.append('stuff to append')
        mock.append.assert_called_once_with('stuff to append')

    def test_order(self):
        my_number = 10

        @callbacks.on_start
        def set_one(state):
            nonlocal my_number
            my_number = 1

        set_one.on_end = Mock()

        @callbacks.on_start
        def set_two(state):
            nonlocal my_number
            my_number = 2

        set_two.on_end = Mock()

        injection = CallbackListInjection(set_one, callbacks.CallbackList([set_two]))

        injection.on_end({})
        self.assertEqual(set_one.on_end.call_count, 1)
        self.assertEqual(set_two.on_end.call_count, 1)

        injection.on_start({})
        self.assertEqual(my_number, 2)


class TestWithGenerators(TestCase):
    def test_with_train_generator_state_filled(self):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_train_generator(generator, 1)

        self.assertTrue(torchbearertrial.state[tb.TRAIN_GENERATOR] == generator)
        self.assertTrue(torchbearertrial.state[tb.TRAIN_STEPS] == 1)

    @patch('warnings.warn')
    def test_with_train_generator_too_many_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_train_generator(generator, 10)

        self.assertTrue(torchbearertrial.state[tb.TRAIN_STEPS] == 2)

    @patch('warnings.warn')
    def test_with_train_generator_fractional_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_train_generator(generator, 1.5)

        self.assertTrue(torchbearertrial.state[tb.TRAIN_STEPS] == 1)

    @patch('warnings.warn')
    def test_with_train_generator_negative_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_train_generator(generator, -2)

        self.assertTrue(torchbearertrial.state[tb.TRAIN_STEPS] == -2)

    @patch('warnings.warn')
    def test_with_train_generator_none_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_train_generator(generator, None)

        self.assertTrue(torchbearertrial.state[tb.TRAIN_STEPS] == 2)

    def test_with_val_generator_state_filled(self):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_val_generator(generator, 1)

        self.assertTrue(torchbearertrial.state[tb.VALIDATION_GENERATOR] == generator)
        self.assertTrue(torchbearertrial.state[tb.VALIDATION_STEPS] == 1)

    @patch('warnings.warn')
    def test_with_val_generator_too_many_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_val_generator(generator, 10)

        self.assertTrue(torchbearertrial.state[tb.VALIDATION_STEPS] == 2)

    @patch('warnings.warn')
    def test_with_val_generator_fractional_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_val_generator(generator, 1.5)

        self.assertTrue(torchbearertrial.state[tb.VALIDATION_STEPS] == 1)

    @patch('warnings.warn')
    def test_with_val_generator_negative_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_val_generator(generator, -2)

        self.assertTrue(torchbearertrial.state[tb.VALIDATION_STEPS] == -2)

    @patch('warnings.warn')
    def test_with_val_generator_none_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_val_generator(generator, None)

        self.assertTrue(torchbearertrial.state[tb.VALIDATION_STEPS] == 2)

    def test_with_test_generator_state_filled(self):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_test_generator(generator, 1)

        self.assertTrue(torchbearertrial.state[tb.TEST_GENERATOR] == generator)
        self.assertTrue(torchbearertrial.state[tb.TEST_STEPS] == 1)

    @patch('warnings.warn')
    def test_with_test_generator_too_many_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_test_generator(generator, 10)

        self.assertTrue(torchbearertrial.state[tb.TEST_STEPS] == 2)

    @patch('warnings.warn')
    def test_with_test_generator_fractional_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_test_generator(generator, 1.5)

        self.assertTrue(torchbearertrial.state[tb.TEST_STEPS] == 1)

    @patch('warnings.warn')
    def test_with_test_generator_negative_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_test_generator(generator, -2)

        self.assertTrue(torchbearertrial.state[tb.TEST_STEPS] == -2)

    @patch('warnings.warn')
    def test_with_test_generator_none_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_test_generator(generator, None)

        self.assertTrue(torchbearertrial.state[tb.TEST_STEPS] == 2)

    @patch('warnings.warn')
    def test_for_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None

        train_steps = 1

        val_steps = 2

        test_steps = 3

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        trainstep = torchbearertrial.for_train_steps = MagicMock()
        valstep = torchbearertrial.for_val_steps = MagicMock()
        teststep = torchbearertrial.for_test_steps = MagicMock()

        torchbearertrial.for_steps(train_steps, val_steps, test_steps)
        trainstep.assert_called_once_with(train_steps)
        valstep.assert_called_once_with(val_steps)
        teststep.assert_called_once_with(test_steps)

    @patch('warnings.warn')
    def test_with_generators(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None

        train_generator = MagicMock()
        train_generator.__len__.return_value = 2
        train_steps = 1

        val_generator = MagicMock()
        val_generator.__len__.return_value = 3
        val_steps = 2

        test_generator = MagicMock()
        test_generator.__len__.return_value = 4
        test_steps = 3

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        traingen = torchbearertrial.with_train_generator = MagicMock()
        valgen = torchbearertrial.with_val_generator = MagicMock()
        testgen = torchbearertrial.with_test_generator = MagicMock()

        torchbearertrial.with_generators(train_generator, val_generator, test_generator, train_steps, val_steps, test_steps)
        traingen.assert_called_once_with(train_generator, train_steps)
        valgen.assert_called_once_with(val_generator, val_steps)
        testgen.assert_called_once_with(test_generator, test_steps)


class TestWithData(TestCase):
    @patch('torchbearer.trial.TensorDataset')
    @patch('torchbearer.trial.DataLoader')
    def test_with_train_data(self, d, td):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        x = torch.rand(1,5)
        y = torch.rand(1,5)
        d.return_value = -1
        steps = 4
        shuffle = False
        num_workers = 1

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_train_generator = MagicMock()
        torchbearertrial.with_train_data(x, y, 1, shuffle=shuffle, num_workers=num_workers, steps=steps)

        d.assert_called_once_with(ANY, 1, shuffle=shuffle, num_workers=num_workers)
        torchbearertrial.with_train_generator.assert_called_once_with(-1, steps=4)
        td.assert_called_once_with(x,y)

    @patch('torchbearer.trial.TensorDataset')
    @patch('torchbearer.trial.DataLoader')
    def test_with_val_data(self, d, td):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        x = torch.rand(1,5)
        y = torch.rand(1,5)
        d.return_value = -1
        steps = 4
        shuffle = False
        num_workers = 1

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_val_generator = MagicMock()
        torchbearertrial.with_val_data(x, y, 1, shuffle=shuffle, num_workers=num_workers, steps=steps)

        d.assert_called_once_with(ANY, 1, shuffle=shuffle, num_workers=num_workers)
        torchbearertrial.with_val_generator.assert_called_once_with(-1, steps=4)
        td.assert_called_once_with(x,y)

    @patch('torchbearer.trial.TensorDataset')
    @patch('torchbearer.trial.DataLoader')
    def test_with_test_data(self, d, td):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        x = torch.rand(1,5)
        y = torch.rand(1,5)
        d.return_value = -1
        steps = 4
        num_workers = 1

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_test_generator = MagicMock()
        torchbearertrial.with_test_data(x, 1, num_workers=num_workers, steps=steps)

        d.assert_called_once_with(ANY, 1, num_workers=num_workers)
        torchbearertrial.with_test_generator.assert_called_once_with(-1, steps=4)
        td.assert_called_once_with(x)


class TestRun(TestCase):
    def test_run_callback_calls(self):
        metric = Metric('test')
        metric.process = Mock(return_value={'test': 0})
        metric.process_final = Mock(return_value={'test': 0})
        metric.reset = Mock(return_value=None)

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)

        epochs = 1

        callback = MagicMock()
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback], pass_state=False)
        torchbearertrial._fit_pass = Mock(return_value={tb.METRICS: {}})
        torchbearertrial._validation_pass = Mock(return_value={tb.METRICS: {}})
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.run(epochs=epochs, verbose=0)

        self.assertEqual(callback.on_start.call_count, 1)
        self.assertEqual(callback.on_start_epoch.call_count, 1)
        self.assertEqual(callback.on_end_epoch.call_count, 1)
        self.assertEqual(callback.on_end.call_count, 1)

    def test_run_epochs_ran_normal(self):
        metric = Metric('test')
        metric.process = Mock(return_value={'test': 0})
        metric.process_final = Mock(return_value={'test': 0})
        metric.reset = Mock(return_value=None)

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)

        epochs = 4

        callback = MagicMock()
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback], pass_state=False)
        torchbearertrial._fit_pass = Mock(return_value={tb.METRICS: {}})
        torchbearertrial._validation_pass = Mock(return_value={tb.METRICS: {}})
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.run(epochs=epochs, verbose=0)

        self.assertTrue(torchbearertrial._fit_pass.call_count == epochs)

    def test_run_epochs_ran_negative(self):
        metric = Metric('test')
        metric.process = Mock(return_value={'test': 0})
        metric.process_final = Mock(return_value={'test': 0})
        metric.reset = Mock(return_value=None)

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)

        epochs = -1

        callback = MagicMock()
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback], pass_state=False)
        torchbearertrial._fit_pass = Mock()
        torchbearertrial._validation_pass = Mock()
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.run(epochs=epochs, verbose=0)

        self.assertTrue(torchbearertrial._fit_pass.call_count == 0)

    def test_run_epochs_history_populated(self):
        metric = Metric('test')
        metric.process = Mock(return_value={'test': 0})
        metric.process_final = Mock(return_value={'test': 0})
        metric.reset = Mock(return_value=None)

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)

        epochs = 10

        callback = MagicMock()
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback], pass_state=False)
        torchbearertrial._fit_pass = Mock(return_value={tb.METRICS: {}})
        torchbearertrial._validation_pass = Mock(return_value={tb.METRICS: {}})
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.state[tb.HISTORY] = [1,2,3,4,5]
        torchbearertrial.run(epochs=epochs, verbose=0)

        self.assertTrue(torchbearertrial._fit_pass.call_count == 5)

    def test_run_fit_pass_Args(self):
        metric = Metric('test')
        metric.process = Mock(return_value={'test': 0})
        metric.process_final = Mock(return_value={'test': 0})
        metric.reset = Mock(return_value=None)

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)

        epochs = 1
        torchmodel = 1

        torchbearertrial = Trial(torchmodel, None, None, [], callbacks=[], pass_state=False)
        torchbearertrial._fit_pass = Mock(return_value={tb.METRICS: {}})
        torchbearertrial._validation_pass = Mock(return_value={tb.METRICS: {}})
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.run(epochs=epochs, verbose=0)

        self.assertEqual(torchbearertrial._fit_pass.call_count, 1)

    def test_run_stop_training(self):
        metric = Metric('test')
        metric.process = Mock(return_value={'test': 0})
        metric.process_final = Mock(return_value={'test': 0})
        metric.reset = Mock(return_value=None)

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)

        epochs = 10

        callback = MagicMock()
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback], pass_state=False)
        torchbearertrial._fit_pass = Mock(return_value={tb.METRICS: {}})
        torchbearertrial._validation_pass = Mock(return_value={tb.METRICS: {}})
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.state[tb.STOP_TRAINING] = True
        torchbearertrial.run(epochs=epochs, verbose=0)

        self.assertEqual(callback.on_start_epoch.call_count, 1)
        self.assertTrue(callback.on_end_epoch.call_count == 0)
        self.assertEqual(callback.on_end.call_count, 1)

    def test_run_stop_training_second(self):
        metric = Metric('test')
        metric.process = Mock(return_value={'test': 0})
        metric.process_final = Mock(return_value={'test': 0})
        metric.reset = Mock(return_value=None)

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)

        epochs = 10
        callback = MagicMock()

        @tb.callbacks.on_end_epoch
        def stop_callback(state):
            state[tb.STOP_TRAINING] = True

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[stop_callback, callback], pass_state=False)
        torchbearertrial._fit_pass = Mock(return_value={tb.METRICS: {}})
        torchbearertrial._validation_pass = Mock(return_value={tb.METRICS: {}})
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.run(epochs=epochs, verbose=0)

        self.assertEqual(callback.on_start_epoch.call_count, 1)
        self.assertEqual(callback.on_end_epoch.call_count, 1)
        self.assertEqual(callback.on_end.call_count, 1)

    def test_run_history_metrics(self):
        metric = Metric('test')
        metric.process = Mock(return_value={'test': 0})
        metric.process_final = Mock(return_value={'test': 0})
        metric.reset = Mock(return_value=None)

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)

        epochs = 1

        callback = MagicMock()
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback], pass_state=False)
        torchbearertrial._fit_pass = Mock(return_value={tb.METRICS: {'fit_test': 1}})
        torchbearertrial._validation_pass = Mock(return_value={'val_test': 2})
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        history = torchbearertrial.run(epochs=epochs, verbose=0)
        self.assertDictEqual(history[0][1], {'fit_test': 1, 'val_test': 2})


class TestFitPass(TestCase):
    def test_fit_train_called(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion, tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu', tb.DATA_TYPE: torch.float,
            tb.HISTORY: [], tb.TRAIN_GENERATOR: generator, tb.TRAIN_STEPS: train_steps, tb.EPOCH: 0
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.state = {tb.TRAIN_GENERATOR: generator, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (generator, train_steps)}

        torchbearertrial._fit_pass(state)
        self.assertEqual(torchbearertrial.train.call_count, 1)

    def test_fit_metrics_reset(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion, tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu', tb.DATA_TYPE: torch.float,
            tb.HISTORY: [], tb.TRAIN_GENERATOR: generator, tb.TRAIN_STEPS: train_steps, tb.EPOCH: 0
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.state = {tb.TRAIN_GENERATOR: generator, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (generator, train_steps)}

        torchbearertrial._fit_pass(state)
        self.assertEqual(metric_list.reset.call_count, 1)

    def test_fit_callback_calls(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        optimizer = MagicMock()
        optimizer.step = lambda closure: closure()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion, tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu', tb.DATA_TYPE: torch.float,
            tb.HISTORY: [], tb.TRAIN_GENERATOR: generator, tb.TRAIN_STEPS: train_steps, tb.EPOCH: 0,
            tb.BACKWARD_ARGS: {}
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.state = {tb.TRAIN_GENERATOR: generator, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (generator, train_steps)}

        torchbearertrial._fit_pass(state)
        self.assertEqual(callback_list.on_start_training.call_count, 1)
        self.assertTrue(callback_list.on_sample.call_count == 3)
        self.assertTrue(callback_list.on_forward.call_count == 3)
        self.assertTrue(callback_list.on_criterion.call_count == 3)
        self.assertTrue(callback_list.on_backward.call_count == 3)
        self.assertTrue(callback_list.on_step_training.call_count == 3)
        self.assertEqual(callback_list.on_end_training.call_count, 1)

    def test_fit_optimizer_calls(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        optimizer = MagicMock()
        optimizer.step = Mock(side_effect=lambda closure: closure())

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion, tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu', tb.DATA_TYPE: torch.float,
            tb.HISTORY: [], tb.TRAIN_GENERATOR: generator, tb.TRAIN_STEPS: train_steps, tb.EPOCH: 0,
            tb.BACKWARD_ARGS: {}
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.state = {tb.TRAIN_GENERATOR: generator, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (generator, train_steps)}

        torchbearertrial._fit_pass(state)
        self.assertTrue(optimizer.zero_grad.call_count == 3)
        self.assertTrue(optimizer.step.call_count == 3)

    def test_fit_forward_call_no_state(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        optimizer = MagicMock()
        optimizer.step = lambda closure: closure()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion, tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu', tb.DATA_TYPE: torch.float,
            tb.HISTORY: [], tb.TRAIN_GENERATOR: generator, tb.TRAIN_STEPS: train_steps, tb.EPOCH: 0,
            tb.BACKWARD_ARGS: {}
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {tb.TRAIN_GENERATOR: generator, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (generator, train_steps)}

        torchbearertrial._fit_pass(state)
        self.assertTrue(torchmodel.call_count == 3)
        self.assertTrue(torchmodel.call_args_list[0][0][0].item() == 1)

    def test_fit_forward_call_with_state(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        optimizer = MagicMock()
        optimizer.step = lambda closure: closure()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float,
            tb.HISTORY: [], tb.TRAIN_GENERATOR: generator, tb.TRAIN_STEPS: train_steps, tb.EPOCH: 0,
            tb.BACKWARD_ARGS: {}
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {tb.TRAIN_GENERATOR: generator, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (generator, train_steps)}

        torchbearertrial._fit_pass(state)
        self.assertTrue(torchmodel.call_count == 3)
        self.assertTrue(len(torchmodel.call_args_list[0][1]) == 1)

    def test_fit_criterion(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()
        optimizer.step = lambda closure: closure()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float,
            tb.HISTORY: [], tb.TRAIN_GENERATOR: generator, tb.TRAIN_STEPS: train_steps, tb.EPOCH: 0,
            tb.BACKWARD_ARGS: {}
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {tb.TRAIN_GENERATOR: generator, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (generator, train_steps)}

        torchbearertrial._fit_pass(state)
        self.assertTrue(criterion.call_count == 3)
        self.assertTrue(criterion.call_args_list[0][0][0] == 5)
        self.assertTrue(criterion.call_args_list[0][0][1].item() == 1.0)

    def test_fit_backward(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()
        optimizer.step = lambda closure: closure()

        loss = MagicMock()
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float,
            tb.HISTORY: [], tb.TRAIN_GENERATOR: generator, tb.TRAIN_STEPS: train_steps, tb.EPOCH: 0,
            tb.BACKWARD_ARGS: {}
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {tb.TRAIN_GENERATOR: generator, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (generator, train_steps)}

        torchbearertrial._fit_pass(state)
        self.assertTrue(loss.backward.call_count == 3)

    def test_fit_metrics_process(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float,
            tb.HISTORY: [], tb.TRAIN_GENERATOR: generator, tb.TRAIN_STEPS: train_steps, tb.EPOCH: 0
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {tb.TRAIN_GENERATOR: generator, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (generator, train_steps)}

        torchbearertrial._fit_pass(state)
        self.assertTrue(metric_list.process.call_count == 3)

    def test_fit_metrics_final(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float,
            tb.HISTORY: [], tb.TRAIN_GENERATOR: generator, tb.TRAIN_STEPS: train_steps, tb.EPOCH: 0
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {tb.TRAIN_GENERATOR: generator, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (generator, train_steps)}

        history = torchbearertrial._fit_pass(state)[tb.METRICS]
        self.assertEqual(metric_list.process_final.call_count, 1)
        self.assertTrue(history['test'] == 2)

    def test_fit_stop_training(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: True, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float,
            tb.HISTORY: [], tb.TRAIN_GENERATOR: generator, tb.TRAIN_STEPS: train_steps, tb.EPOCH: 0
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {tb.TRAIN_GENERATOR: generator, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (generator, train_steps)}

        torchbearertrial._fit_pass(state)
        self.assertEqual(metric_list.process.call_count, 1)

    def test_fit_iterator_none(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = 1
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: True, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float, tb.HISTORY: [], tb.TRAIN_GENERATOR: None, tb.TRAIN_STEPS: steps, tb.EPOCH: 0,
            tb.X: data[0][0], tb.Y_TRUE: data[0][1]
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {tb.TRAIN_GENERATOR: None, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (None, steps)}

        state = torchbearertrial._fit_pass(state)
        self.assertTrue(state[tb.ITERATOR] is None)

    def test_fit_state_values(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = 1
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()
        optimizer.step = lambda closure: closure()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: True, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float, tb.HISTORY: [], tb.TRAIN_GENERATOR: generator, tb.TRAIN_STEPS: steps, tb.EPOCH: 0,
            tb.X: data[0][0], tb.Y_TRUE: data[0][1],
            tb.BACKWARD_ARGS: {}
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {tb.TRAIN_GENERATOR: generator, tb.CALLBACK_LIST: callback_list, tb.TRAIN_DATA: (generator, steps)}

        state = torchbearertrial._fit_pass(state)
        self.assertTrue(state[tb.ITERATOR] is not None)
        self.assertTrue(state[tb.Y_PRED] == 5)
        self.assertTrue(state[tb.LOSS].item() == 2)
        self.assertTrue(state[tb.METRICS]['test'] == 2)


class TestTestPass(TestCase):
    def test_metric_reset(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float, tb.HISTORY: [], tb.GENERATOR: generator, tb.STEPS: steps, tb.EPOCH: 0,
            tb.X: data[0][0], tb.Y_TRUE: data[0][1], tb.SAMPLER: tb.trial.Sampler(load_batch_none)
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {tb.GENERATOR: generator, tb.CALLBACK_LIST: callback_list}

        torchbearertrial._test_pass(state)
        self.assertEqual(metric_list.reset.call_count, 1)

    def test_callback_calls(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float, tb.HISTORY: [], tb.GENERATOR: generator, tb.STEPS: steps, tb.EPOCH: 0,
            tb.X: data[0][0], tb.Y_TRUE: data[0][1], tb.SAMPLER: tb.trial.Sampler(load_batch_none)
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {tb.GENERATOR: generator, tb.CALLBACK_LIST: callback_list}

        torchbearertrial._test_pass(state)
        self.assertEqual(callback_list.on_start_validation.call_count, 1)
        self.assertTrue(callback_list.on_sample_validation.call_count == 3)
        self.assertTrue(callback_list.on_forward_validation.call_count == 3)
        self.assertTrue(callback_list.on_criterion_validation.call_count == 3)
        self.assertTrue(callback_list.on_step_validation.call_count == 3)
        self.assertEqual(callback_list.on_end_validation.call_count, 1)

    def test_forward_no_state(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float, tb.HISTORY: [], tb.GENERATOR: generator, tb.STEPS: steps, tb.EPOCH: 0,
            tb.X: data[0][0], tb.Y_TRUE: data[0][1], tb.SAMPLER: tb.trial.Sampler(load_batch_standard)
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {tb.GENERATOR: generator, tb.CALLBACK_LIST: callback_list}

        torchbearertrial._test_pass(state)
        self.assertTrue(torchmodel.call_count == 3)
        self.assertTrue(torchmodel.call_args_list[0][0][0].item() == 1)

    def test_forward_with_state(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float, tb.HISTORY: [], tb.GENERATOR: generator, tb.STEPS: steps, tb.EPOCH: 0,
            tb.X: data[0][0], tb.Y_TRUE: data[0][1], tb.SAMPLER: tb.trial.Sampler(load_batch_none)
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {tb.GENERATOR: generator, tb.CALLBACK_LIST: callback_list}

        torchbearertrial._test_pass(state)
        self.assertTrue(torchmodel.call_count == 3)
        self.assertTrue(len(torchmodel.call_args_list[0][1]) == 1)

    def test_criterion(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float, tb.HISTORY: [], tb.GENERATOR: generator, tb.STEPS: steps, tb.EPOCH: 0,
            tb.X: data[0][0], tb.Y_TRUE: data[0][1], tb.SAMPLER: tb.trial.Sampler(load_batch_standard)
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {tb.GENERATOR: generator, tb.CALLBACK_LIST: callback_list}

        torchbearertrial._test_pass(state)
        self.assertTrue(criterion.call_count == 3)
        self.assertTrue(criterion.call_args_list[0][0][0] == 5)
        self.assertTrue(criterion.call_args_list[0][0][1].item() == 1.0)

    def test_metric_process(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float, tb.HISTORY: [], tb.GENERATOR: generator, tb.STEPS: steps, tb.EPOCH: 0,
            tb.X: data[0][0], tb.Y_TRUE: data[0][1], tb.SAMPLER: tb.trial.Sampler(load_batch_none)
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {tb.GENERATOR: generator, tb.CALLBACK_LIST: callback_list}

        torchbearertrial._test_pass(state)
        self.assertTrue(metric_list.process.call_count == 3)

    def test_metric_final(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: False, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float, tb.HISTORY: [], tb.GENERATOR: generator, tb.STEPS: steps, tb.EPOCH: 0,
            tb.X: data[0][0], tb.Y_TRUE: data[0][1], tb.SAMPLER: tb.trial.Sampler(load_batch_none)
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {tb.GENERATOR: generator, tb.CALLBACK_LIST: callback_list}

        history = torchbearertrial._test_pass(state)
        self.assertEqual(metric_list.process_final.call_count, 1)
        self.assertTrue(history[tb.METRICS]['test'] == 2)

    def test_stop_training(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: True, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float, tb.HISTORY: [], tb.GENERATOR: generator, tb.STEPS: steps, tb.EPOCH: 0,
            tb.X: data[0][0], tb.Y_TRUE: data[0][1], tb.SAMPLER: tb.trial.Sampler(load_batch_none)
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {tb.GENERATOR: generator, tb.CALLBACK_LIST: callback_list}

        torchbearertrial._test_pass(state)
        self.assertEqual(metric_list.process.call_count, 1)

    def test_iterator_none(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = 1
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: True, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float, tb.HISTORY: [], tb.GENERATOR: None, tb.STEPS: steps, tb.EPOCH: 0,
            tb.X: data[0][0], tb.Y_TRUE: data[0][1], tb.SAMPLER: tb.trial.Sampler(load_batch_none)
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {tb.GENERATOR: None, tb.CALLBACK_LIST: callback_list}

        state = torchbearertrial._test_pass(state)
        self.assertTrue(state[tb.ITERATOR] is None)

    def test_state_values(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = 1
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        tb.CallbackListInjection = Mock(return_value=callback_list)

        state = {
            tb.MAX_EPOCHS: epochs, tb.STOP_TRAINING: True, tb.MODEL: torchmodel, tb.CRITERION: criterion,
            tb.OPTIMIZER: optimizer,
            tb.METRIC_LIST: metric_list, tb.CALLBACK_LIST: callback_list, tb.DEVICE: 'cpu',
            tb.DATA_TYPE: torch.float, tb.HISTORY: [], tb.GENERATOR: generator, tb.STEPS: steps, tb.EPOCH: 0,
            tb.X: data[0][0], tb.Y_TRUE: data[0][1], tb.SAMPLER: tb.trial.Sampler(load_batch_none)
        }

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[], pass_state=False)
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {tb.GENERATOR: generator, tb.CALLBACK_LIST: callback_list}

        state = torchbearertrial._test_pass(state)
        self.assertTrue(state[tb.ITERATOR] is not None)
        self.assertTrue(state[tb.Y_PRED] == 5)
        self.assertTrue(state[tb.LOSS].item() == 2)
        self.assertTrue(state[tb.METRICS]['test'] == 2)


class TestTrialValEvalPred(TestCase):
    def test_validation_pass(self):
        generator = MagicMock()
        steps = 5
        tb.CallbackListInjection = Mock()

        state = {tb.VALIDATION_GENERATOR: generator, tb.VALIDATION_STEPS: steps, tb.METRICS: 1}
        t = Trial(MagicMock())
        eval_mock = t.eval = Mock()
        test_pass_mock = t._test_pass = Mock()
        t.state = {tb.VALIDATION_GENERATOR: generator, tb.CALLBACK_LIST: None, tb.VALIDATION_DATA: (generator, steps)}
        metrics = t._validation_pass(state)

        self.assertEqual(eval_mock.call_count, 1)
        self.assertEqual(test_pass_mock.call_count, 1)
        test_pass_state = test_pass_mock.call_args[0][0]
        self.assertTrue(test_pass_state[tb.GENERATOR] == generator)
        self.assertTrue(test_pass_state[tb.STEPS] == steps)
        self.assertTrue(metrics == 1)

    def test_validation_pass_none(self):
        generator = None
        steps = None
        tb.CallbackListInjection = Mock()

        state = {tb.VALIDATION_GENERATOR: generator, tb.VALIDATION_STEPS: steps, tb.METRICS: 1}
        t = Trial(MagicMock())
        eval_mock = t.eval = Mock()
        t._test_pass = Mock()
        t.state = {tb.VALIDATION_GENERATOR: generator, tb.CALLBACK_LIST: None, tb.VALIDATION_DATA: (generator, steps)}
        t._validation_pass(state)

        self.assertTrue(eval_mock.call_count == 0)
        
    def test_evaluate(self):
        generator = MagicMock()
        steps = 5
        tb.CallbackListInjection = Mock()

        t = Trial(MagicMock())
        eval_mock = t.eval = Mock()
        clist = MagicMock()
        state = {tb.HISTORY: [('steps', {'train_metric': 2})], tb.VALIDATION_GENERATOR: generator, tb.CALLBACK_LIST: clist, tb.VALIDATION_STEPS: steps, tb.VALIDATION_DATA: (generator, steps), tb.METRICS: {'val_metric': 1}}
        test_pass_mock = t._test_pass = Mock(return_value=state)
        t.state = state
        metrics = t.evaluate()

        self.assertEqual(clist.on_start.call_count, 1)
        self.assertEqual(clist.on_start_epoch.call_count, 1)
        self.assertEqual(clist.on_end_epoch.call_count, 1)
        self.assertEqual(clist.on_end.call_count, 1)
        self.assertEqual(eval_mock.call_count, 1)
        self.assertEqual(test_pass_mock.call_count, 1)
        test_pass_state = test_pass_mock.call_args[0][0]
        self.assertTrue(test_pass_state[tb.GENERATOR] == generator)
        self.assertTrue(test_pass_state[tb.STEPS] == steps)
        self.assertEqual(metrics['val_metric'], 1)
        self.assertDictEqual(state[tb.HISTORY][0][1], {'train_metric': 2, 'val_metric': 1})

    def test_evaluate_none(self):
        generator = None
        steps = None
        tb.CallbackListInjection = Mock()

        t = Trial(MagicMock())
        eval_mock = t.eval = Mock()
        test_pass_mock = t._test_pass = Mock(return_value={tb.METRICS: 1})
        t.state = {tb.VALIDATION_GENERATOR: generator, tb.CALLBACK_LIST: None, tb.VALIDATION_STEPS: steps, tb.VALIDATION_DATA: (generator, steps)}
        metrics = t.evaluate()

        self.assertTrue(eval_mock.call_count == 0)

    def test_predict(self):
        generator = MagicMock()
        steps = 5
        tb.CallbackListInjection = Mock()

        state = {tb.TEST_GENERATOR: generator, tb.TEST_STEPS: steps, tb.METRICS: 1}
        t = Trial(MagicMock())
        eval_mock = t.eval = Mock()
        test_pass_mock = t._test_pass = Mock(return_value={tb.FINAL_PREDICTIONS: 1})
        clist = MagicMock()
        t.state = {tb.TEST_GENERATOR: generator, tb.CALLBACK_LIST: clist, tb.TEST_STEPS: steps, tb.TEST_DATA: (generator, steps)}
        metrics = t.predict(state)

        self.assertEqual(clist.on_start.call_count, 1)
        self.assertEqual(clist.on_start_epoch.call_count, 1)
        self.assertEqual(clist.on_end_epoch.call_count, 1)
        self.assertEqual(clist.on_end.call_count, 1)
        self.assertEqual(eval_mock.call_count, 1)
        self.assertEqual(test_pass_mock.call_count, 1)
        test_pass_state = test_pass_mock.call_args[0][0]
        self.assertTrue(test_pass_state[tb.GENERATOR] == generator)
        self.assertTrue(test_pass_state[tb.STEPS] == steps)
        self.assertTrue(metrics == 1)

    def test_predict_none(self):
        generator = None
        steps = None
        tb.CallbackListInjection = Mock()

        state = {tb.TEST_GENERATOR: generator, tb.TEST_STEPS: steps, tb.METRICS: 1}
        t = Trial(MagicMock())
        eval_mock = t.eval = Mock()
        test_pass_mock = t._test_pass = Mock(return_value={tb.FINAL_PREDICTIONS: 1})
        t.state = {tb.TEST_GENERATOR: generator, tb.CALLBACK_LIST: None, tb.TEST_STEPS: steps, tb.TEST_DATA: (generator, steps)}
        metrics = t.predict(state)

        self.assertTrue(eval_mock.call_count == 0)


class TestReplay(TestCase):
    @patch('torchbearer.trial.Tqdm')
    def test_replay_tqdm(self, tq):
        t = Trial(MagicMock())
        callback = MagicMock()
        history = [((10, 5), {'test': i, 'val_test2': i+1}) for i in range(10)]

        t.state[tb.HISTORY] = history
        t.replay(callbacks=[callback])
        self.assertEqual(tq.call_count, 1)

    @patch('torchbearer.trial.Tqdm')
    def test_replay_no_tqdm(self, tq):
        t = Trial(MagicMock())
        callback = MagicMock()
        history = [((10, 5), {'test': i, 'val_test2': i+1}) for i in range(10)]

        t.state[tb.HISTORY] = history
        t.replay(callbacks=[callback], verbose=0)
        tq.assert_not_called()

    def test_replay_callback_calls(self):
        t = Trial(MagicMock())
        callback = MagicMock()
        history = [((10, 5), {'test': i, 'val_test2': i+1}) for i in range(10)]

        t.state[tb.HISTORY] = history
        t.replay(callbacks=[callback], verbose=0)
        self.assertEqual(callback.on_start.call_count, 1)
        self.assertTrue(callback.on_sample.call_count == 100)
        self.assertTrue(callback.on_sample_validation.call_count == 50)

    def test_replay_one_batch_true(self):
        t = Trial(MagicMock())
        callback = MagicMock()
        history = [((10, 5), {'test': i, 'val_test2': i+1}) for i in range(1)]

        t.state[tb.HISTORY] = history
        t.replay(callbacks=[callback], verbose=0, one_batch=True)
        self.assertEqual(callback.on_start.call_count, 1)
        self.assertTrue(callback.on_sample.call_count == 1)
        self.assertTrue(callback.on_sample_validation.call_count == 1)

    def test_replay_metrics(self):
        t = Trial(MagicMock())
        callback = MagicMock()
        history = [((10, 5), {'test': i, 'val_test2': i+1}) for i in range(10)]

        t.state[tb.HISTORY] = history
        t.replay(callbacks=[callback], verbose=0)

        self.assertTrue(callback.on_sample.call_args_list[0][0][0][tb.METRICS]['test'] == 9)
        self.assertTrue(callback.on_sample_validation.call_args_list[0][0][0][tb.METRICS]['val_test2'] == 10)

    def test_replay_stop_training(self):
        t = Trial(MagicMock())
        callback = MagicMock()

        @tb.callbacks.on_sample
        def stop_training(state):
            state[tb.STOP_TRAINING] = True

        history = [((10, 5), {'test': i, 'val_test2': i+1}) for i in range(10)]

        t.state[tb.HISTORY] = history
        t.replay(callbacks=[callback, stop_training], verbose=0)

        self.assertTrue(callback.on_sample.call_count == 10)
        callback.on_sample_validation.assert_not_called()


class TestTrialMembers(TestCase):
    def test_init_none_criterion(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        optimizer = MagicMock()
        metric = MagicMock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [metric], [], pass_state=False).to('cpu', torch.float64)
        loss = torchbearertrial.state[tb.CRITERION](None, None)
        self.assertTrue(str(loss.device) == 'cpu')
        self.assertTrue(loss.dtype == torch.float64)
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(loss.shape == torch.Size([1]))
        self.assertTrue(loss.item() == 0)

    def test_init_none_criterion_add(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        optimizer = MagicMock()
        metric = MagicMock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [metric], [], pass_state=False).to('cpu', torch.float64)
        loss = torchbearertrial.state[tb.CRITERION](None, None)
        loss = loss + 1
        self.assertTrue(str(loss.device) == 'cpu')
        self.assertTrue(loss.dtype == torch.float64)
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(loss.shape == torch.Size([1]))
        self.assertTrue(loss.item() == 1)

    def test_str(self):
        torchmodel = "mod"
        optimizer = "opt"
        metric = tb.metrics.Metric('met')

        torchbearertrial = Trial(torchmodel, optimizer, "crit", [metric], ["cb"], pass_state=False)
        correct_string = "--------------------- OPTIMZER ---------------------\nopt\n\n-------------------- CRITERION ---------------------\ncrit\n\n--------------------- METRICS ----------------------\n['met']\n\n-------------------- CALLBACKS ---------------------\n['cb']\n\n---------------------- MODEL -----------------------\nmod\n\n"
        self.assertEqual(str(torchbearertrial), correct_string)

    def test_repr(self):
        torchmodel = "mod"
        optimizer = "opt"
        metric = tb.metrics.Metric('met')

        torchbearertrial = Trial(torchmodel, optimizer, "crit", [metric], ["cb"], pass_state=False)
        self.assertEqual(str(torchbearertrial), repr(torchbearertrial))

    def test_train(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        optimizer = MagicMock()
        metric = MagicMock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [metric], [], pass_state=False)
        torchbearertrial.train()
        self.assertTrue(torchbearertrial.state[tb.MODEL].training == True)
        self.assertEqual(metric.train.call_count, 1)

    def test_eval(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        optimizer = MagicMock()
        metric = MagicMock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [metric], [], pass_state=False)
        torchbearertrial.eval()
        self.assertTrue(torchbearertrial.state[tb.MODEL].training == False)
        self.assertEqual(metric.eval.call_count, 1)

    def test_to_both_args(self):
        dev = 'cuda:1'
        dtype = torch.float16

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.to = Mock()
        optimizer = torch.optim.Adam(torchmodel.parameters(), 0.1)
        state_tensor = torch.Tensor([1])
        state_tensor.to = Mock()
        optimizer.state = {'test': {'test': state_tensor}}

        torchbearertrial = Trial(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearertrial.to(dev, dtype)

        self.assertTrue(torchmodel.to.call_args[0][0] == dev)
        self.assertTrue(torchmodel.to.call_args[0][1] == dtype)
        self.assertTrue(state_tensor.to.call_args[0][0] == dev)
        self.assertTrue(state_tensor.to.call_args[0][1] == dtype)

    def test_to_only_device(self):
        dev = 'cuda:1'

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.to = Mock()
        optimizer = torch.optim.Adam(torchmodel.parameters(), 0.1)
        state_tensor = torch.Tensor([1])
        state_tensor.to = Mock()
        optimizer.state = {'test': {'test': state_tensor}}

        torchbearertrial = Trial(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearertrial.to(dev)

        self.assertTrue(torchmodel.to.call_args[0][0] == dev)
        self.assertTrue(state_tensor.to.call_args[0][0] == dev)

    def test_to_only_dtype(self):
        dtype = torch.float16

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.to = Mock()
        optimizer = torch.optim.Adam(torchmodel.parameters(), 0.1)
        state_tensor = torch.Tensor([1])
        state_tensor.to = Mock()
        optimizer.state = {'test': {'test': state_tensor}}

        torchbearertrial = Trial(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearertrial.to(dtype)

        self.assertTrue(torchmodel.to.call_args[0][0] == dtype)
        self.assertTrue(state_tensor.to.call_args[0][0] == dtype)

    def test_to_kwargs(self):
        dev = 'cuda:1'
        dtype = torch.float16

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.to = Mock()
        optimizer = torch.optim.Adam(torchmodel.parameters(), 0.1)
        state_tensor = torch.Tensor([1])
        state_tensor.to = Mock()
        optimizer.state = {'test': {'test': state_tensor}}

        torchbearertrial = Trial(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearertrial.to(device=dev, dtype=dtype)

        self.assertTrue(torchmodel.to.call_args[1]['device'] == dev)
        self.assertTrue(torchmodel.to.call_args[1]['dtype'] == dtype)
        self.assertTrue(state_tensor.to.call_args[1]['device'] == dev)
        self.assertTrue(state_tensor.to.call_args[1]['dtype'] == dtype)
        
    @patch('torch.cuda.current_device')
    def test_cuda_no_device(self, device_mock):
        device_mock.return_value = 111

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        torchbearertrial = Trial(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearertrial.to = Mock()
        torchbearertrial.cuda()

        self.assertTrue(torchbearertrial.to.call_args[0][0] == 'cuda:' + str(111))

    def test_cuda_with_device(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        torchbearertrial = Trial(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearertrial.to = Mock()
        torchbearertrial.cuda(device='2')

        self.assertTrue(torchbearertrial.to.call_args[0][0] == 'cuda:2')

    def test_cpu(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        torchbearertrial = Trial(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearertrial.to = Mock()
        torchbearertrial.cpu()

        self.assertTrue(torchbearertrial.to.call_args[0][0] == 'cpu')
        
    def test_load_state_dict_resume(self):
        key_words = {'strict': True}

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()
        torch_state = torchmodel.state_dict()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()
        optimizer_state = optimizer.state_dict()

        callback_list = MagicMock()
        callback_list.state_dict = Mock(return_value = 1)

        history = ['test']

        torchbearertrial = Trial(torchmodel, optimizer, None, [], [], pass_state=False)
        torchbearertrial.state[tb.CALLBACK_LIST] = callback_list
        torchbearertrial.state[tb.HISTORY] = history
        torchbearer_state = torchbearertrial.state_dict()
        torchbearertrial.state[tb.HISTORY] = 'Wrong'

        torchbearertrial.load_state_dict(torchbearer_state, **key_words)

        self.assertTrue(torchmodel.load_state_dict.call_args[0][0] == torch_state)
        self.assertTrue(optimizer.load_state_dict.call_args[0][0] == optimizer_state)
        self.assertTrue(optimizer.load_state_dict.call_args[0][0] == optimizer_state)
        self.assertTrue(callback_list.load_state_dict.call_args[0][0] == 1)

        self.assertTrue(torchbearertrial.state[tb.HISTORY] == history)
        self.assertEqual(torchbearertrial.state[tb.MODEL].load_state_dict.call_count, 1)
        self.assertEqual(torchbearertrial.state[tb.OPTIMIZER].load_state_dict.call_count, 1)
        self.assertEqual(torchbearertrial.state[tb.CALLBACK_LIST].load_state_dict.call_count, 1)
        self.assertTrue(torchmodel.load_state_dict.call_args[1] == key_words)

    def test_load_state_dict_no_resume(self):
        key_words = {'strict': True}

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()
        torch_state = torchmodel.state_dict()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()
        optimizer_state = optimizer.state_dict()

        history = ['test']

        torchbearertrial = Trial(torchmodel, optimizer, None, [], [], pass_state=False)
        torchbearertrial.state[tb.HISTORY] = history
        torchbearer_state = torchbearertrial.state_dict()
        torchbearertrial.state[tb.HISTORY] = 'Wrong'

        torchbearertrial.load_state_dict(torchbearer_state, resume=False, **key_words)

        self.assertTrue(torchbearertrial.state[tb.HISTORY] is 'Wrong')
        self.assertEqual(torchbearertrial.state[tb.MODEL].load_state_dict.call_count, 1)
        self.assertTrue(torchbearertrial.state[tb.OPTIMIZER].load_state_dict.call_count == 0)
        self.assertTrue(torchmodel.load_state_dict.call_args[1] == key_words)

    def test_load_state_dict_wrong_version(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1, 1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [], [], pass_state=False)

        torchbearer_state = torchbearertrial.state_dict()
        torchbearer_state[tb.VERSION] = '0.1.7'  # Old version

        self.assertWarns(UserWarning, lambda: torchbearertrial.load_state_dict(torchbearer_state, resume=True))

    def test_load_state_dict_not_torchbearer(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1, 1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [], [], pass_state=False)

        torchbearer_state = torchbearertrial.state_dict()
        torchbearer_state[tb.VERSION] = '0.1.7'  # Old version

        self.assertWarns(UserWarning, lambda: torchbearertrial.load_state_dict(torchbearer_state[tb.MODEL]))

        self.assertEqual(torchmodel.load_state_dict.call_count, 1)
        optimizer.load_state_dict.assert_not_called()

    def test_state_dict(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel_state = torchmodel.state_dict()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer_state = optimizer.state_dict()

        callback_list = MagicMock()
        callback_list.state_dict = Mock(return_value = 1)

        history = ['test']

        torchbearertrial = Trial(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearertrial.state[tb.HISTORY] = history
        torchbearertrial.state[tb.CALLBACK_LIST] = callback_list
        torchbearer_state = torchbearertrial.state_dict()

        self.assertTrue(torchbearer_state[tb.VERSION] == tb.__version__.replace('.dev', ''))
        self.assertTrue(torchbearer_state[tb.MODEL] == torchmodel_state)
        self.assertTrue(torchbearer_state[tb.OPTIMIZER] == optimizer_state)
        self.assertTrue(torchbearer_state[tb.CALLBACK_LIST] == 1)
        self.assertTrue(torchbearer_state[tb.HISTORY] == history)

    def test_state_dict_kwargs(self):
        keywords = {'destination': None, 'prefix': '', 'keep_vars': False}
        torchmodel = MagicMock()
        optimizer = MagicMock()

        torchbearertrial = Trial(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearertrial.state_dict(**keywords)

        self.assertTrue(torchmodel.state_dict.call_args[1] == keywords)
        self.assertTrue(optimizer.state_dict.call_args[1] == {})


class TestTrialFunctions(TestCase):
    @patch('torchbearer.trial.Tqdm')
    def test_get_printer_no_tqdm(self, tq):
        verbose = 0
        validation_label_letter = 'v'

        printer = tb.trial.get_printer(verbose=verbose, validation_label_letter=validation_label_letter)
        tq.assert_not_called()

    @patch('torchbearer.trial.Tqdm')
    def test_get_printer_verbose_1(self, tq):
        verbose = 1
        validation_label_letter = 'v'

        printer = tb.trial.get_printer(verbose=verbose, validation_label_letter=validation_label_letter)
        tq.assert_called_once_with(on_epoch=True, validation_label_letter=validation_label_letter)

    @patch('torchbearer.trial.Tqdm')
    def test_get_printer_verbose_2(self, tq):
        verbose = 2
        validation_label_letter = 'v'

        printer = tb.trial.get_printer(verbose=verbose, validation_label_letter=validation_label_letter)
        tq.assert_called_once_with(validation_label_letter=validation_label_letter)

    @patch('torchbearer.trial.Tqdm')
    def test_get_printer_letter(self, tq):
        verbose = 2
        validation_label_letter = 'r'

        printer = tb.trial.get_printer(verbose=verbose, validation_label_letter=validation_label_letter)
        tq.assert_called_once_with(validation_label_letter=validation_label_letter)

    @patch('torchbearer.trial.get_printer')
    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_printer_no_tqdm(self, c_inj, get_print_mock):
        callback_list = tb.callbacks.CallbackList([])

        class SomeClass:
            @tb.inject_printer('v')
            def test_func(self, verbose=0):
                pass

        t = SomeClass()
        t.state = {tb.CALLBACK_LIST: callback_list}
        t.test_func(verbose=0)
        self.assertEqual(c_inj.call_count, 1)
        get_print_mock.assert_called_once_with(validation_label_letter='v', verbose=0)

    @patch('torchbearer.trial.get_printer')
    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_printer_tqdm_on_epoch(self, c_inj, get_print_mock):
        callback_list = tb.callbacks.CallbackList([])

        class SomeClass:
            @tb.inject_printer('t')
            def test_func(self, verbose=0):
                pass

        t = SomeClass()
        t.state = {tb.CALLBACK_LIST: callback_list}
        t.test_func(verbose=1)
        self.assertEqual(c_inj.call_count, 1)
        get_print_mock.assert_called_once_with(validation_label_letter='t', verbose=1)

    @patch('torchbearer.trial.get_printer')
    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_printer_tqdm_on_batch(self, c_inj, get_print_mock):
        callback_list = tb.callbacks.CallbackList([])

        class SomeClass:
            @tb.inject_printer('t')
            def test_func(self, verbose=0):
                pass

        t = SomeClass()
        t.state = {tb.CALLBACK_LIST: callback_list}
        t.test_func(verbose=2)
        self.assertEqual(c_inj.call_count, 1)
        get_print_mock.assert_called_once_with(validation_label_letter='t', verbose=2)

    @patch('torchbearer.trial.get_printer')
    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_printer_tqdm_default(self, c_inj, get_print_mock):
        callback_list = tb.callbacks.CallbackList([])

        class SomeClass:
            @tb.inject_printer('t')
            def test_func(self, verbose=2):
                pass

        t = SomeClass()
        t.state = {tb.CALLBACK_LIST: callback_list}
        t.test_func()
        self.assertEqual(c_inj.call_count, 1)
        get_print_mock.assert_called_once_with(validation_label_letter='t', verbose=2)

    @patch('torchbearer.trial.Tqdm')
    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_printer_injection(self, c_inj, tq):
        callback_list = tb.callbacks.CallbackList([])

        class SomeClass:
            @tb.inject_printer('v')
            def test_func(self_inner, verbose=0):
                self.assertEqual(c_inj.call_count, 1)

        t = SomeClass()
        t.state = {tb.CALLBACK_LIST: callback_list}
        t.test_func()
        self.assertTrue(t.state[tb.CALLBACK_LIST] == callback_list)

    def test_inject_sampler_standard(self):
        generator = MagicMock()
        steps = None

        class SomeClass:
            @tb.inject_sampler(tb.GENERATOR)
            def test_func(self):
                pass

        t = SomeClass()
        t.state = {tb.GENERATOR: (generator, steps)}
        t.test_func()
        self.assertTrue(t.state[tb.SAMPLER].batch_loader == tb.trial.load_batch_standard)

    def test_inject_sampler_none(self):
        generator = None
        steps = None

        class SomeClass:
            @tb.inject_sampler(tb.GENERATOR)
            def test_func(self):
                pass

        t = SomeClass()
        t.state = {tb.GENERATOR: (generator, steps)}
        t.test_func()
        self.assertTrue(t.state[tb.SAMPLER].batch_loader == tb.trial.load_batch_none)

    def test_inject_sampler_predict(self):
        generator = MagicMock()
        steps = None

        class SomeClass:
            @tb.inject_sampler(tb.GENERATOR, predict=True)
            def test_func(self):
                pass

        t = SomeClass()
        t.state = {tb.GENERATOR: (generator, steps)}
        t.test_func()
        self.assertTrue(t.state[tb.SAMPLER].batch_loader == tb.trial.load_batch_predict)

    def test_inject_sampler_data_key(self):
        generator = MagicMock()
        test_generator = 'test'
        test_steps = 1

        class SomeClass:
            @tb.inject_sampler(tb.GENERATOR, predict=False)
            def test_func(self, data_key=None):
                pass

        t = SomeClass()
        t.state = {tb.GENERATOR: (generator, None), tb.TEST_GENERATOR: (test_generator, test_steps)}
        t.test_func(data_key=tb.TEST_GENERATOR)
        self.assertTrue(t.state[tb.GENERATOR] == test_generator)
        self.assertTrue(t.state[tb.STEPS] == test_steps)

    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_callback(self, c_inj):
        callback_list = tb.callbacks.CallbackList([])
        test_callback = MagicMock()

        class SomeClass:
            @tb.inject_callback(test_callback)
            def test_func(self_inner):
                self.assertEqual(c_inj.call_count, 1)

        t = SomeClass()
        t.state = {tb.CALLBACK_LIST: callback_list}
        t.test_func()
        self.assertTrue(c_inj.call_args[0][0] == test_callback)

    def test_deep_to_tensor(self):
        tensor = MagicMock()
        new_dtype = torch.float16
        new_device = 'cuda:1'

        deep_to(tensor, new_device, new_dtype)
        self.assertTrue(tensor.to.call_args[0][0] == new_device)
        self.assertTrue(tensor.to.call_args[0][1] == new_dtype)

    def test_deep_to_tensor_int_dtype(self):
        tensor = MagicMock()
        tensor.dtype = torch.uint8
        new_device = 'cuda:1'
        new_dtype = torch.uint8

        deep_to(tensor, new_device, new_dtype)
        self.assertTrue(tensor.to.call_args[0][0] == new_device)
        self.assertTrue(len(tensor.to.call_args[0]) == 1)

    def test_deep_to_list(self):
        tensor_1 = MagicMock()
        tensor_2 = MagicMock()
        tensors = [tensor_1, tensor_2]
        new_dtype = torch.float16
        new_device = 'cuda:1'

        deep_to(tensors, new_device, new_dtype)
        for tensor in tensors:
            self.assertTrue(tensor.to.call_args[0][0] == new_device)
            self.assertTrue(tensor.to.call_args[0][1] == new_dtype)

    def test_deep_to_dict(self):
        tensor_1 = torch.Tensor([0])
        tensor_1.to = Mock()
        tensor_2 = torch.Tensor([0])
        tensor_2.to = Mock()
        tensors = {'t1': tensor_1, 't2': tensor_2}
        new_dtype = torch.float16
        new_device = 'cuda:1'

        deep_to(tensors, new_device, new_dtype)
        self.assertTrue(tensor_1.to.call_args[0][0] == new_device)
        self.assertTrue(tensor_1.to.call_args[0][1] == new_dtype)
        self.assertTrue(tensor_2.to.call_args[0][0] == new_device)
        self.assertTrue(tensor_2.to.call_args[0][1] == new_dtype)

    def test_load_batch_standard(self):
        items = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2]))]
        iterator = iter(items)

        state = {tb.ITERATOR: iterator, tb.DEVICE: 'cpu', tb.DATA_TYPE: torch.int}

        load_batch_standard(state)
        self.assertTrue(state[tb.X].item() == items[0][0].item())
        self.assertTrue(state[tb.Y_TRUE].item() == items[0][1].item())

    def test_load_batch_none(self):
        items = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2]))]
        iterator = iter(items)

        state = {tb.ITERATOR: iterator, tb.DEVICE: 'cpu', tb.DATA_TYPE: torch.int}

        load_batch_none(state)
        self.assertTrue(state[tb.X] is None)
        self.assertTrue(state[tb.Y_TRUE] is None)

    def test_load_batch_predict_data(self):
        items = [torch.Tensor([1]), torch.Tensor([2])]
        iterator = iter(items)

        state = {tb.ITERATOR: iterator, tb.DEVICE: 'cpu', tb.DATA_TYPE: torch.int}
        load_batch_predict(state)
        self.assertTrue(state[tb.X].item() == items[0].item())

    def test_load_batch_predict_list(self):
        items = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2]))]
        iterator = iter(items)

        state = {tb.ITERATOR: iterator, tb.DEVICE: 'cpu', tb.DATA_TYPE: torch.int}

        load_batch_predict(state)
        self.assertTrue(state[tb.X].item() == items[0][0].item())
        self.assertTrue(state[tb.Y_TRUE].item() == items[0][1].item())

    def test_update_device_and_dtype_only_kwarg(self):
        main_state = {}
        dtype = torch.float16
        dev = 'cuda:1'

        kwargs = {str(tb.DEVICE): dev, str(tb.DATA_TYPE): dtype}

        main_state = update_device_and_dtype(main_state, **kwargs)

        self.assertTrue(main_state[tb.DATA_TYPE] == dtype)
        self.assertTrue(main_state[tb.DEVICE] == dev)

    def test_update_device_and_dtype_only_arg(self):
        main_state = {}
        dtype = torch.float16
        dev = 'cuda:1'
        args = (dtype, dev)

        main_state = update_device_and_dtype(main_state, *args)

        self.assertTrue(main_state[tb.DATA_TYPE] == dtype)
        self.assertTrue(main_state[tb.DEVICE] == dev)
