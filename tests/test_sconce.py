from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch
import torch
from torch.utils.data import DataLoader

from sconce import Model
import sconce
from sconce.callbacks import Callback
from sconce.metrics import MetricList, Metric


class TestSconce(TestCase):

    def test_main_loop_metrics(self):
        metric = Metric('test')
        metric.process = Mock(return_value=0)
        metric.process_final = Mock(return_value=0)
        metric.reset = Mock(return_value=None)
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)

        epochs = 1

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2], requires_grad=True)
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        sconcestate[sconce.METRIC_LIST].metric_list[0].reset.assert_called_once()
        self.assertTrue(sconcestate[sconce.METRIC_LIST].metric_list[0].process.call_count == len(data))
        sconcestate[sconce.METRIC_LIST].metric_list[0].process_final.assert_called_once()
        self.assertTrue(sconcestate[sconce.METRICS]['test'] == 0)

    def test_main_loop_train_steps_positive(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = 2

        epochs = 1

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2], requires_grad=True)
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == train_steps)

    @patch("warnings.warn")
    def test_main_loop_train_steps_fractional(self, _):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = 2.5

        epochs = 1

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2], requires_grad=True)
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == int(train_steps))

    def test_main_loop_epochs_positive(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 2

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2], requires_grad=True)
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == len(data)*epochs)

    def test_main_loop_epochs_zero(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 0

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2], requires_grad=True)
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == len(data)*epochs)

    def test_main_loop_epochs_negative(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = -2

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2], requires_grad=True)
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == 0)

    @patch("warnings.warn")
    def test_main_loop_epochs_fractional(self, _):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 2.5

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2], requires_grad=True)
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == int(epochs)*len(data))

    def test_main_loop_train_steps_too_big(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = 8

        epochs = 1

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2], requires_grad=True)
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == len(data))

    def test_main_loop_train_steps_negative(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = -2

        epochs = 1

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2], requires_grad=True)
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == 0)

    def test_main_loop_pass_state(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 1

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2], requires_grad=True)
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=True)

        self.assertTrue(len(sconcestate[sconce.MODEL].call_args) == 2)

    def test_main_loop_optimizer(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 1

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2], requires_grad=True)
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=True)
        self.assertTrue(optimizer.zero_grad.call_count == epochs*len(data))
        self.assertTrue(optimizer.step.call_count == epochs*len(data))

    def test_main_loop_criterion(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 1

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2], requires_grad=True)
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])

        state = sconcemodel.main_state.copy()
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=True)
        self.assertTrue(sconcestate[sconce.CRITERION].call_count == epochs*len(data))
        self.assertTrue(sconcestate[sconce.LOSS] == 2)

    def test_main_loop_backward(self):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 1

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = Mock()
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=True)
        self.assertTrue(sconcestate[sconce.LOSS].backward.call_count == epochs*len(data))

    def test_main_loop_stop_training(self):
        class stop_training_test_callback(Callback):
            def on_sample(self, state):
                super().on_sample(state)
                state[sconce.STOP_TRAINING] = True

        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 1

        callback = stop_training_test_callback()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = Mock()
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=True)
        self.assertTrue(sconcestate[sconce.MODEL].call_count == 1)

    def test_main_loop_callback_calls(self):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = 2

        epochs = 1

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = Mock()
        criterion = Mock(return_value=loss)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])
        sconcestate = sconcemodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=True)
        callback.on_start.assert_called_once()
        callback.on_start_epoch.asser_called_once()
        callback.on_start_training.assert_called_once()
        self.assertTrue(callback.on_sample.call_count == train_steps*epochs)
        self.assertTrue(callback.on_forward.call_count == train_steps*epochs)
        self.assertTrue(callback.on_criterion.call_count == train_steps*epochs)
        self.assertTrue(callback.on_backward.call_count == train_steps*epochs)
        self.assertTrue(callback.on_step_training.call_count == train_steps*epochs)
        callback.on_end_training.assert_called_once()
        callback.on_end_epoch.assert_called_once()


    def test_test_loop_metrics(self):
        metric = Metric('test')
        metric.process = Mock(return_value=0)
        metric.process_final = Mock(return_value=0)
        metric.reset = Mock(return_value=None)
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = len(data)

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])

        state = sconcemodel.main_state.copy()
        state.update({sconce.METRIC_LIST: metric_list, sconce.VALIDATION_GENERATOR: validation_generator,
                 sconce.CallbackList: callback_List, sconce.MODEL: torchmodel, sconce.VALIDATION_STEPS: validation_steps,
                 sconce.CRITERION: criterion, sconce.STOP_TRAINING: False, sconce.METRICS: {}})

        sconcestate = sconcemodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=None)

        sconcestate[sconce.METRIC_LIST].metric_list[0].reset.assert_called_once()
        self.assertTrue(sconcestate[sconce.METRIC_LIST].metric_list[0].process.call_count == len(data))
        sconcestate[sconce.METRIC_LIST].metric_list[0].process_final.assert_called_once()
        self.assertTrue(sconcestate[sconce.METRICS]['test'] == 0)

    def test_test_loop_forward(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = len(data)

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])

        state = sconcemodel.main_state.copy()
        state.update({sconce.METRIC_LIST: metric_list, sconce.VALIDATION_GENERATOR: validation_generator,
                 sconce.CallbackList: callback_List, sconce.VALIDATION_STEPS: validation_steps,
                 sconce.CRITERION: criterion, sconce.STOP_TRAINING: False, sconce.METRICS: {}})

        sconcestate = sconcemodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=None)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == 3)
        self.assertTrue(sconcestate[sconce.Y_PRED] == 1)

    def test_test_loop_criterion(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = len(data)

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])

        state = sconcemodel.main_state.copy()
        state.update({sconce.METRIC_LIST: metric_list, sconce.VALIDATION_GENERATOR: validation_generator,
                 sconce.CallbackList: callback_List, sconce.VALIDATION_STEPS: validation_steps,
                 sconce.CRITERION: criterion, sconce.STOP_TRAINING: False, sconce.METRICS: {}})

        sconcestate = sconcemodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=None)

        self.assertTrue(sconcestate[sconce.CRITERION].call_count == 3)
        self.assertTrue(sconcestate[sconce.LOSS] == 2)

    def test_test_loop_pass_state(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = len(data)

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])

        state = sconcemodel.main_state.copy()
        state.update({sconce.METRIC_LIST: metric_list, sconce.VALIDATION_GENERATOR: validation_generator,
                      sconce.CallbackList: callback_List, sconce.VALIDATION_STEPS: validation_steps,
                      sconce.CRITERION: criterion, sconce.STOP_TRAINING: False, sconce.METRICS: {}})

        sconcestate = sconcemodel._test_loop(state, callback_List, True, Model._load_batch_standard, num_steps=None)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == 3)
        self.assertTrue(len(sconcestate[sconce.MODEL].call_args) == 2)

    def test_test_loop_num_steps_positive(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = 2

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])

        state = sconcemodel.main_state.copy()
        state.update({sconce.METRIC_LIST: metric_list, sconce.VALIDATION_GENERATOR: validation_generator,
                      sconce.CallbackList: callback_List, sconce.VALIDATION_STEPS: validation_steps,
                      sconce.CRITERION: criterion, sconce.STOP_TRAINING: False, sconce.METRICS: {}})

        sconcestate = sconcemodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=validation_steps)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == 2)

    @patch("warnings.warn")
    def test_test_loop_num_steps_fractional(self, _):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = 2.5

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])

        state = sconcemodel.main_state.copy()
        state.update({sconce.METRIC_LIST: metric_list, sconce.VALIDATION_GENERATOR: validation_generator,
                      sconce.CallbackList: callback_List, sconce.VALIDATION_STEPS: validation_steps,
                      sconce.CRITERION: criterion, sconce.STOP_TRAINING: False, sconce.METRICS: {}})

        sconcestate = sconcemodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=validation_steps)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == 2)

    def test_test_loop_num_steps_too_big(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = 8

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])

        state = sconcemodel.main_state.copy()
        state.update({sconce.METRIC_LIST: metric_list, sconce.VALIDATION_GENERATOR: validation_generator,
                      sconce.CallbackList: callback_List, sconce.VALIDATION_STEPS: validation_steps,
                      sconce.CRITERION: criterion, sconce.STOP_TRAINING: False, sconce.METRICS: {}})

        sconcestate = sconcemodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=validation_steps)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == len(data))

    def test_test_loop_num_steps_zero(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = 0

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])

        state = sconcemodel.main_state.copy()
        state.update({sconce.METRIC_LIST: metric_list, sconce.VALIDATION_GENERATOR: validation_generator,
                      sconce.CallbackList: callback_List, sconce.VALIDATION_STEPS: validation_steps,
                      sconce.CRITERION: criterion, sconce.STOP_TRAINING: False, sconce.METRICS: {}})

        sconcestate = sconcemodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=validation_steps)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == 0)

    def test_test_loop_num_steps_negative(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = -2

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])

        state = sconcemodel.main_state.copy()
        state.update({sconce.METRIC_LIST: metric_list, sconce.VALIDATION_GENERATOR: validation_generator,
                      sconce.CallbackList: callback_List, sconce.VALIDATION_STEPS: validation_steps,
                      sconce.CRITERION: criterion, sconce.STOP_TRAINING: False, sconce.METRICS: {}})

        sconcestate = sconcemodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=validation_steps)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == 0)

    def test_test_loop_stop_training(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = len(data)

        callback = MagicMock()
        callback_List = sconce.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        sconcemodel = Model(torchmodel, optimizer, criterion, [metric])

        state = sconcemodel.main_state.copy()
        state.update({sconce.METRIC_LIST: metric_list, sconce.VALIDATION_GENERATOR: validation_generator,
                      sconce.CallbackList: callback_List, sconce.VALIDATION_STEPS: validation_steps,
                      sconce.CRITERION: criterion, sconce.STOP_TRAINING: True, sconce.METRICS: {}})

        sconcestate = sconcemodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=None)

        self.assertTrue(sconcestate[sconce.MODEL].call_count == 1)

    def test_to_both_args(self):
        dev = 'cuda:1'
        dtype = torch.float16

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.to = Mock()
        optimizer = torch.optim.Adam(torchmodel.parameters(), 0.1)
        state_tensor = torch.Tensor([1])
        state_tensor.to = Mock()
        optimizer.state = {'test': {'test': state_tensor}}

        sconcemodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        sconcemodel.to(dev, dtype)

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

        sconcemodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        sconcemodel.to(dev)

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

        sconcemodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        sconcemodel.to(dtype)

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

        sconcemodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        sconcemodel.to(device=dev, dtype=dtype)

        self.assertTrue(torchmodel.to.call_args[1]['device'] == dev)
        self.assertTrue(torchmodel.to.call_args[1]['dtype'] == dtype)
        self.assertTrue(state_tensor.to.call_args[1]['device'] == dev)
        self.assertTrue(state_tensor.to.call_args[1]['dtype'] == dtype)

    def test_update_device_and_dtype_from_args_only_kwarg(self):
        main_state = {}
        dtype = torch.float16
        dev = 'cuda:1'
        kwargs = {sconce.DEVICE: dev, sconce.DATA_TYPE: dtype}

        main_state = Model._update_device_and_dtype_from_args(main_state, **kwargs)

        self.assertTrue(main_state[sconce.DATA_TYPE] == dtype)
        self.assertTrue(main_state[sconce.DEVICE] == dev)

    def test_update_device_and_dtype_from_args_only_arg(self):
        main_state = {}
        dtype = torch.float16
        dev = 'cuda:1'
        args = (dtype, dev)

        main_state = Model._update_device_and_dtype_from_args(main_state, *args)

        self.assertTrue(main_state[sconce.DATA_TYPE] == dtype)
        self.assertTrue(main_state[sconce.DEVICE] == dev)

    @patch('torch.cuda.current_device')
    def test_cuda_no_device(self, device_mock):
        device_mock.return_value = 111

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        sconcemodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        sconcemodel.to = Mock()
        sconcemodel.cuda()

        self.assertTrue(sconcemodel.to.call_args[0][0] == 'cuda:' + str(111))

    def test_cuda_with_device(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        sconcemodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        sconcemodel.to = Mock()
        sconcemodel.cuda(device='2')

        self.assertTrue(sconcemodel.to.call_args[0][0] == 'cuda:2')

    def test_cpu(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        sconcemodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        sconcemodel.to = Mock()
        sconcemodel.cpu()

        self.assertTrue(sconcemodel.to.call_args[0][0] == 'cpu')

    def test_load_state_dict(self):
        key_words = {'strict': True}

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()
        torch_state = torchmodel.state_dict()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()
        optimizer_state = optimizer.state_dict()

        sconcemodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        sconce_state = sconcemodel.state_dict()

        sconcemodel.load_state_dict(sconce_state, **key_words)

        self.assertTrue(torchmodel.load_state_dict.call_args[0][0] == torch_state)
        self.assertTrue(optimizer.load_state_dict.call_args[0][0] == optimizer_state)
        self.assertTrue(torchmodel.load_state_dict.call_args[1] == key_words)

    def test_state_dict(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel_state = torchmodel.state_dict()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer_state = optimizer.state_dict()

        sconcemodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        sconce_state = sconcemodel.state_dict()

        self.assertTrue(sconce_state[sconce.MODEL] == torchmodel_state)
        self.assertTrue(sconce_state[sconce.OPTIMIZER] == optimizer_state)

    def test_state_dict_kwargs(self):
        keywords = {'destination': None, 'prefix': '', 'keep_vars': False}
        torchmodel = MagicMock()
        optimizer = MagicMock()

        sconcemodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        sconcemodel.state_dict(**keywords)

        self.assertTrue(torchmodel.state_dict.call_args[1] == keywords)
        self.assertTrue(optimizer.state_dict.call_args[1] == {})

    def test_deep_to_tensor(self):
        tensor = MagicMock()
        new_dtype = torch.float16
        new_device = 'cuda:1'

        Model._deep_to(tensor, new_device, new_dtype)
        self.assertTrue(tensor.to.call_args[0][0] == new_device)
        self.assertTrue(tensor.to.call_args[0][1] == new_dtype)

    def test_deep_to_tensor_int_dtype(self):
        tensor = MagicMock()
        tensor.dtype = torch.uint8
        new_device = 'cuda:1'
        new_dtype = torch.uint8

        Model._deep_to(tensor, new_device, new_dtype)
        self.assertTrue(tensor.to.call_args[0][0] == new_device)
        self.assertTrue(len(tensor.to.call_args[0]) == 1)

    def test_deep_to_list(self):
        tensor_1 = MagicMock()
        tensor_2 = MagicMock()
        tensors = [tensor_1, tensor_2]
        new_dtype = torch.float16
        new_device = 'cuda:1'

        Model._deep_to(tensors, new_device, new_dtype)
        for tensor in tensors:
            self.assertTrue(tensor.to.call_args[0][0] == new_device)
            self.assertTrue(tensor.to.call_args[0][1] == new_dtype)

    def test_load_batch_standard(self):
        items = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2]))]
        iterator = iter(items)
        state = {'training_iterator': iterator, 'device': 'cpu', 'dtype': torch.int}

        Model._load_batch_standard('training', state)
        self.assertTrue(state['x'].item() == items[0][0].item())
        self.assertTrue(state['y_true'].item() == items[0][1].item())

    def test_load_batch_predict_data(self):
        items = [torch.Tensor([1]), torch.Tensor([2])]
        iterator = iter(items)
        state = {'training_iterator': iterator, 'device': 'cpu', 'dtype': torch.int}
        Model._load_batch_predict('training', state)
        self.assertTrue(state['x'].item() == items[0].item())

    def test_load_batch_predict_list(self):
        items = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2]))]
        iterator = iter(items)
        state = {'training_iterator': iterator, 'device': 'cpu', 'dtype': torch.int}

        Model._load_batch_predict('training', state)
        self.assertTrue(state['x'].item() == items[0][0].item())
        self.assertTrue(state['y_true'].item() == items[0][1].item())
