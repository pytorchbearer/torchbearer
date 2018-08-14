from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch, ANY
import torch
from torch.utils.data import DataLoader

from torchbearer import Trial
import torchbearer as tb
from torchbearer.callbacks import Callback
from torchbearer.metrics import MetricList, Metric


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
        torchbearertrial._fit_pass = Mock()
        torchbearertrial._validation_pass = Mock()
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.run(epochs=epochs, verbose=0)

        callback.on_start.assert_called_once()
        callback.on_start_epoch.assert_called_once()
        callback.on_end_epoch.assert_called_once()
        callback.on_end.assert_called_once()

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
        torchbearertrial._fit_pass = Mock()
        torchbearertrial._validation_pass = Mock()
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
        torchbearertrial._fit_pass = Mock()
        torchbearertrial._validation_pass = Mock()
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
        torchbearertrial._fit_pass = Mock()
        torchbearertrial._validation_pass = Mock()
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.run(epochs=epochs, verbose=0)

        torchbearertrial._fit_pass.assert_called_once()

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
        torchbearertrial._fit_pass = Mock()
        torchbearertrial._validation_pass = Mock()
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.state[tb.STOP_TRAINING] = True
        torchbearertrial.run(epochs=epochs, verbose=0)

        callback.on_start_epoch.assert_called_once()
        self.assertTrue(callback.on_end_epoch.call_count == 0)
        callback.on_end.assert_called_once()

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
        torchbearertrial._fit_pass = Mock()
        torchbearertrial._validation_pass = Mock()
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.run(epochs=epochs, verbose=0)

        callback.on_start_epoch.assert_called_once()
        callback.on_end_epoch.assert_called_once()
        callback.on_end.assert_called_once()

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
        torchbearertrial._fit_pass = Mock(return_value={'fit_test': 1})
        torchbearertrial._validation_pass = Mock(return_value={'val_test': 2})
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        history = torchbearertrial.run(epochs=epochs, verbose=0)
        self.assertDictEqual(history[0], {'fit_test': 1, 'val_test': 2})


# class TestFitPass(TestCase):
#     def test_state_additions(self):
#         metric = Metric('test')
#         metric.process = Mock(return_value={'test': 0})
#         metric.process_final = Mock(return_value={'test': 0})
#         metric.reset = Mock(return_value=None)
#
#         data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
#         generator = DataLoader(data)
#         train_steps = len(data)
#         epochs = 1
#         callback = MagicMock()
#         torchmodel = MagicMock()
#         torchmodel.forward = Mock(return_value=1)
#         optimizer = MagicMock()
#
#         loss = torch.tensor([2.0], requires_grad=True)
#         criterion = Mock(return_value=loss)
#
#         torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback], pass_state=False)
#         torchbearertrial.with_train_generator(generator, steps=train_steps)
#         torchbearertrial.run(epochs=epochs, verbose=0)


class TestTrialMembers(TestCase):
    def test_train(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        optimizer = MagicMock()
        metric = MagicMock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [metric], [], pass_state=False)
        torchbearertrial.train()
        self.assertTrue(torchbearertrial.state[tb.MODEL].training == True)
        metric.train.assert_called_once()

    def test_eval(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        optimizer = MagicMock()
        metric = MagicMock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [metric], [], pass_state=False)
        torchbearertrial.eval()
        self.assertTrue(torchbearertrial.state[tb.MODEL].training == False)
        metric.eval.assert_called_once()

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

        history = ['test']

        torchbearertrial = Trial(torchmodel, optimizer, None, [], [], pass_state=False)
        torchbearertrial.state[tb.HISTORY] = history
        torchbearer_state = torchbearertrial.state_dict()
        torchbearertrial.state[tb.HISTORY] = 'Wrong'

        torchbearertrial.load_state_dict(torchbearer_state, **key_words)

        self.assertTrue(torchmodel.load_state_dict.call_args[0][0] == torch_state)
        self.assertTrue(optimizer.load_state_dict.call_args[0][0] == optimizer_state)
        self.assertTrue(optimizer.load_state_dict.call_args[0][0] == optimizer_state)
        self.assertTrue(torchbearertrial.state[tb.HISTORY] == history)
        torchbearertrial.state[tb.MODEL].load_state_dict.assert_called_once()
        torchbearertrial.state[tb.OPTIMIZER].load_state_dict.assert_called_once()
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
        torchbearertrial.state[tb.MODEL].load_state_dict.assert_called_once()
        self.assertTrue(torchbearertrial.state[tb.OPTIMIZER].load_state_dict.call_count == 0)
        self.assertTrue(torchmodel.load_state_dict.call_args[1] == key_words)

    def test_state_dict(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel_state = torchmodel.state_dict()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer_state = optimizer.state_dict()

        history = ['test']

        torchbearertrial = Trial(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearertrial.state[tb.HISTORY] = history
        torchbearer_state = torchbearertrial.state_dict()

        self.assertTrue(torchbearer_state[tb.MODEL] == torchmodel_state)
        self.assertTrue(torchbearer_state[tb.OPTIMIZER] == optimizer_state)
        self.assertTrue(torchbearer_state[tb.HISTORY] == history)

    def test_state_dict_kwargs(self):
        keywords = {'destination': None, 'prefix': '', 'keep_vars': False}
        torchmodel = MagicMock()
        optimizer = MagicMock()

        torchbearertrial = Trial(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearertrial.state_dict(**keywords)

        self.assertTrue(torchmodel.state_dict.call_args[1] == keywords)
        self.assertTrue(optimizer.state_dict.call_args[1] == {})
