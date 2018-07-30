from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch
import torch
from torch.utils.data import DataLoader

from torchbearer import Model
import torchbearer
from torchbearer.callbacks import Callback
from torchbearer.metrics import MetricList, Metric


class TestTorchbearer(TestCase):
    @patch('torchbearer.cv_utils.get_train_valid_sets')
    def test_fit_valid_sets_args(self, gtvs):
        x = torch.rand(1,5)
        y = torch.rand(1,5)
        val_data = (1,2)
        val_split = 0.2
        shuffle = False

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()
        metric = Metric('test')

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        gtvs.return_value = (1, 2)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearermodel.fit_generator = Mock()
        torchbearermodel.fit(x, y, 1, validation_data=val_data, validation_split=val_split, shuffle=shuffle)

        gtvs.assert_called_once()
        self.assertTrue(list(gtvs.call_args[0][0].numpy()[0]) == list(x.numpy()[0]))
        self.assertTrue(list(gtvs.call_args[0][1].numpy()[0]) == list(y.numpy()[0]))
        self.assertTrue(gtvs.call_args[0][2] == val_data)
        self.assertTrue(gtvs.call_args[0][3] == val_split)
        self.assertTrue(gtvs.call_args[1]['shuffle'] == shuffle)

    def test_fit_no_valid(self):
        x = torch.rand(1, 5)
        y = torch.rand(1, 5)

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()
        metric = Metric('test')

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearermodel.fit_generator = Mock()
        fit = torchbearermodel.fit_generator
        torchbearermodel.fit(x, y, 1, validation_split=None)

        self.assertTrue(fit.call_args[1]['validation_generator'] is None)

    def test_main_loop_metrics(self):
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

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        torchbearerstate[torchbearer.METRIC_LIST].metric_list[0].reset.assert_called_once()
        self.assertTrue(torchbearerstate[torchbearer.METRIC_LIST].metric_list[0].process.call_count == len(data))
        torchbearerstate[torchbearer.METRIC_LIST].metric_list[0].process_final.assert_called_once()
        self.assertTrue(torchbearerstate[torchbearer.METRICS]['test'] == 0)

    def test_main_loop_verbose(self):
        metric = Metric('test')

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

        import sys
        from io import StringIO
        saved_std_err = sys.stderr
        out = StringIO()
        sys.stderr = out

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 1, [callback], initial_epoch=0, pass_state=False)

        output = out.getvalue().strip()
        self.assertTrue(output != '')
        sys.stderr = saved_std_err

    def test_main_loop_train_steps_positive(self):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = 2

        epochs = 1

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == train_steps)

    @patch("warnings.warn")
    def test_main_loop_train_steps_fractional(self, _):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = 2.5

        epochs = 1

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == int(train_steps))

    def test_main_loop_validation_setup(self):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        valgenerator = DataLoader(data)
        train_steps = 2

        epochs = 1

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearermodel._test_loop = Mock()
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback],
                                                          validation_generator=valgenerator, initial_epoch=0,
                                                          pass_state=False)

        self.assertTrue(torchbearerstate[torchbearer.VALIDATION_STEPS] == len(valgenerator))
        self.assertTrue(torchbearerstate[torchbearer.VALIDATION_GENERATOR] == valgenerator)

    def test_main_loop_epochs_positive(self):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 2

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == len(data)*epochs)

    def test_main_loop_epochs_zero(self):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 0

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == len(data)*epochs)

    def test_main_loop_epochs_negative(self):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = -2

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == 0)

    @patch("warnings.warn")
    def test_main_loop_epochs_fractional(self, _):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 2.5

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == int(epochs)*len(data))

    @patch("warnings.warn")
    def test_main_loop_epochs_none(self, warning):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = None

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(warning.call_count == 1)

    def test_main_loop_none_gen(self):
        metric = Metric('test')

        generator = None
        train_steps = 8

        epochs = 1

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == train_steps)

    def test_main_loop_train_steps_too_big(self):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = 8

        epochs = 1

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == len(data))

    def test_main_loop_train_steps_negative(self):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = -2

        epochs = 1

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=False)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == 0)

    def test_main_loop_pass_state(self):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 1

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=True)

        self.assertTrue(len(torchbearerstate[torchbearer.MODEL].call_args) == 2)

    def test_main_loop_optimizer(self):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 1

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=True)
        self.assertTrue(optimizer.zero_grad.call_count == epochs*len(data))
        self.assertTrue(optimizer.step.call_count == epochs*len(data))

    def test_main_loop_criterion(self):
        metric = Metric('test')

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = None

        epochs = 1

        callback = MagicMock()

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])

        state = torchbearermodel.main_state.copy()
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=True)
        self.assertTrue(torchbearerstate[torchbearer.CRITERION].call_count == epochs*len(data))
        self.assertTrue(torchbearerstate[torchbearer.LOSS] == 2)

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

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=True)
        self.assertTrue(torchbearerstate[torchbearer.LOSS].backward.call_count == epochs*len(data))

    def test_main_loop_stop_training(self):
        class stop_training_test_callback(Callback):
            def on_sample(self, state):
                super().on_sample(state)
                state[torchbearer.STOP_TRAINING] = True

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

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=True)
        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == 1)

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

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearerstate = torchbearermodel.fit_generator(generator, train_steps, epochs, 0, [callback], initial_epoch=0, pass_state=True)
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
        metric.process = Mock(return_value={'test': 0})
        metric.process_final = Mock(return_value={'test': 0})
        metric.reset = Mock(return_value=None)
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = len(data)

        callback = MagicMock()
        callback_List = torchbearer.CallbackList([callback])

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])

        state = torchbearermodel.main_state.copy()
        state.update({torchbearer.METRIC_LIST: metric_list, torchbearer.VALIDATION_GENERATOR: validation_generator,
                 torchbearer.CallbackList: callback_List, torchbearer.MODEL: torchmodel, torchbearer.VALIDATION_STEPS: validation_steps,
                 torchbearer.CRITERION: criterion, torchbearer.STOP_TRAINING: False, torchbearer.METRICS: {}})

        torchbearerstate = torchbearermodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=None)

        torchbearerstate[torchbearer.METRIC_LIST].metric_list[0].reset.assert_called_once()
        self.assertTrue(torchbearerstate[torchbearer.METRIC_LIST].metric_list[0].process.call_count == len(data))
        torchbearerstate[torchbearer.METRIC_LIST].metric_list[0].process_final.assert_called_once()
        self.assertTrue(torchbearerstate[torchbearer.METRICS]['test'] == 0)

    def test_test_loop_forward(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = len(data)

        callback = MagicMock()
        callback_List = torchbearer.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])

        state = torchbearermodel.main_state.copy()
        state.update({torchbearer.METRIC_LIST: metric_list, torchbearer.VALIDATION_GENERATOR: validation_generator,
                 torchbearer.CallbackList: callback_List, torchbearer.VALIDATION_STEPS: validation_steps,
                 torchbearer.CRITERION: criterion, torchbearer.STOP_TRAINING: False, torchbearer.METRICS: {}})

        torchbearerstate = torchbearermodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=None)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == 3)
        self.assertTrue(torchbearerstate[torchbearer.Y_PRED] == 1)

    def test_test_loop_criterion(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = len(data)

        callback = MagicMock()
        callback_List = torchbearer.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])

        state = torchbearermodel.main_state.copy()
        state.update({torchbearer.METRIC_LIST: metric_list, torchbearer.VALIDATION_GENERATOR: validation_generator,
                 torchbearer.CallbackList: callback_List, torchbearer.VALIDATION_STEPS: validation_steps,
                 torchbearer.CRITERION: criterion, torchbearer.STOP_TRAINING: False, torchbearer.METRICS: {}})

        torchbearerstate = torchbearermodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=None)

        self.assertTrue(torchbearerstate[torchbearer.CRITERION].call_count == 3)
        self.assertTrue(torchbearerstate[torchbearer.LOSS] == 2)

    def test_test_loop_pass_state(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = len(data)

        callback = MagicMock()
        callback_List = torchbearer.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])

        state = torchbearermodel.main_state.copy()
        state.update({torchbearer.METRIC_LIST: metric_list, torchbearer.VALIDATION_GENERATOR: validation_generator,
                      torchbearer.CallbackList: callback_List, torchbearer.VALIDATION_STEPS: validation_steps,
                      torchbearer.CRITERION: criterion, torchbearer.STOP_TRAINING: False, torchbearer.METRICS: {}})

        torchbearerstate = torchbearermodel._test_loop(state, callback_List, True, Model._load_batch_standard, num_steps=None)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == 3)
        self.assertTrue(len(torchbearerstate[torchbearer.MODEL].call_args) == 2)

    def test_test_loop_num_steps_positive(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = 2

        callback = MagicMock()
        callback_List = torchbearer.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])

        state = torchbearermodel.main_state.copy()
        state.update({torchbearer.METRIC_LIST: metric_list, torchbearer.VALIDATION_GENERATOR: validation_generator,
                      torchbearer.CallbackList: callback_List, torchbearer.VALIDATION_STEPS: validation_steps,
                      torchbearer.CRITERION: criterion, torchbearer.STOP_TRAINING: False, torchbearer.METRICS: {}})

        torchbearerstate = torchbearermodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=validation_steps)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == 2)

    @patch("warnings.warn")
    def test_test_loop_num_steps_fractional(self, _):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = 2.5

        callback = MagicMock()
        callback_List = torchbearer.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])

        state = torchbearermodel.main_state.copy()
        state.update({torchbearer.METRIC_LIST: metric_list, torchbearer.VALIDATION_GENERATOR: validation_generator,
                      torchbearer.CallbackList: callback_List, torchbearer.VALIDATION_STEPS: validation_steps,
                      torchbearer.CRITERION: criterion, torchbearer.STOP_TRAINING: False, torchbearer.METRICS: {}})

        torchbearerstate = torchbearermodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=validation_steps)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == 2)

    def test_test_loop_num_steps_too_big(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = 8

        callback = MagicMock()
        callback_List = torchbearer.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])

        state = torchbearermodel.main_state.copy()
        state.update({torchbearer.METRIC_LIST: metric_list, torchbearer.VALIDATION_GENERATOR: validation_generator,
                      torchbearer.CallbackList: callback_List, torchbearer.VALIDATION_STEPS: validation_steps,
                      torchbearer.CRITERION: criterion, torchbearer.STOP_TRAINING: False, torchbearer.METRICS: {}})

        torchbearerstate = torchbearermodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=validation_steps)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == len(data))

    def test_test_loop_num_steps_zero(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = 0

        callback = MagicMock()
        callback_List = torchbearer.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])

        state = torchbearermodel.main_state.copy()
        state.update({torchbearer.METRIC_LIST: metric_list, torchbearer.VALIDATION_GENERATOR: validation_generator,
                      torchbearer.CallbackList: callback_List, torchbearer.VALIDATION_STEPS: validation_steps,
                      torchbearer.CRITERION: criterion, torchbearer.STOP_TRAINING: False, torchbearer.METRICS: {}})

        torchbearerstate = torchbearermodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=validation_steps)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == 0)

    def test_test_loop_num_steps_negative(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = -2

        callback = MagicMock()
        callback_List = torchbearer.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])

        state = torchbearermodel.main_state.copy()
        state.update({torchbearer.METRIC_LIST: metric_list, torchbearer.VALIDATION_GENERATOR: validation_generator,
                      torchbearer.CallbackList: callback_List, torchbearer.VALIDATION_STEPS: validation_steps,
                      torchbearer.CRITERION: criterion, torchbearer.STOP_TRAINING: False, torchbearer.METRICS: {}})

        torchbearerstate = torchbearermodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=validation_steps)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == 0)

    def test_test_loop_stop_training(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        validation_generator = DataLoader(data)
        validation_steps = len(data)

        callback = MagicMock()
        callback_List = torchbearer.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])

        state = torchbearermodel.main_state.copy()
        state.update({torchbearer.METRIC_LIST: metric_list, torchbearer.VALIDATION_GENERATOR: validation_generator,
                      torchbearer.CallbackList: callback_List, torchbearer.VALIDATION_STEPS: validation_steps,
                      torchbearer.CRITERION: criterion, torchbearer.STOP_TRAINING: True, torchbearer.METRICS: {}})

        torchbearerstate = torchbearermodel._test_loop(state, callback_List, False, Model._load_batch_standard, num_steps=None)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == 1)

    def test_test_loop_none_gen(self):
        metric = Metric('test')
        metric_list = MetricList([metric])

        validation_generator = None
        validation_steps = 8

        callback = MagicMock()
        callback_List = torchbearer.CallbackList([callback])

        torchmodel = Mock(return_value=1)
        optimizer = MagicMock()

        criterion = Mock(return_value=2)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])

        state = torchbearermodel.main_state.copy()
        state.update({torchbearer.METRIC_LIST: metric_list, torchbearer.VALIDATION_GENERATOR: validation_generator,
                      torchbearer.CallbackList: callback_List, torchbearer.VALIDATION_STEPS: validation_steps,
                      torchbearer.CRITERION: criterion, torchbearer.STOP_TRAINING: False, torchbearer.METRICS: {}})

        torchbearerstate = torchbearermodel._test_loop(state, callback_List, False, Model._load_batch_none, num_steps=validation_steps)

        self.assertTrue(torchbearerstate[torchbearer.MODEL].call_count == validation_steps)

    def test_evaluate(self):
        x = torch.rand(1,5)
        y = torch.rand(1,5)
        pass_state = False
        verbose=0

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()
        metric = Metric('test')

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearermodel.evaluate_generator = Mock()
        ev = torchbearermodel.evaluate_generator
        torchbearermodel.evaluate(x, y, verbose=verbose, pass_state=pass_state)

        ev.assert_called_once()
        self.assertTrue(ev.call_args[0][1] == verbose)
        self.assertTrue(ev.call_args[1]['pass_state'] == pass_state)

    def test_evaluate_generator_args(self):
        torchmodel = MagicMock()
        optimizer = MagicMock()
        generator = MagicMock()

        pass_state = False
        steps = None

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.main_state[torchbearer.METRICS] = 1
        torchbearermodel._test_loop = Mock()

        torchbearermodel.evaluate_generator(generator, 0, steps, pass_state)
        self.assertTrue(torchbearermodel._test_loop.call_args[0][1].callback_list == [])
        self.assertTrue(torchbearermodel._test_loop.call_args[0][2] == pass_state)
        self.assertTrue(torchbearermodel._test_loop.call_args[0][4] == steps)

    def test_evaluate_generator_none(self):
        torchmodel = MagicMock()
        optimizer = MagicMock()
        generator = None

        pass_state = False
        steps = 10

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.main_state[torchbearer.METRICS] = 1
        torchbearermodel._test_loop = Mock()

        torchbearermodel.evaluate_generator(generator, 0, steps, pass_state)
        self.assertTrue(torchbearermodel._test_loop.call_args[0][1].callback_list == [])
        self.assertTrue(torchbearermodel._test_loop.call_args[0][2] == pass_state)
        self.assertTrue(torchbearermodel._test_loop.call_args[0][4] == steps)

    def test_evaluate_generator_verbose(self):
        from torchbearer.callbacks import Tqdm

        torchmodel = MagicMock()
        optimizer = MagicMock()
        generator = MagicMock()

        pass_state = False
        steps = None

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.main_state[torchbearer.METRICS] = 1
        torchbearermodel._test_loop = Mock()

        torchbearermodel.evaluate_generator(generator, 1, steps, pass_state)
        self.assertIsInstance(torchbearermodel._test_loop.call_args[0][1].callback_list[0], Tqdm)

    def test_evaluate_generator_pass_state(self):
        torchmodel = MagicMock()
        optimizer = MagicMock()
        generator = MagicMock()

        pass_state = True
        steps = None

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.main_state[torchbearer.METRICS] = 1
        torchbearermodel._test_loop = Mock()

        torchbearermodel.evaluate_generator(generator, 0, steps, pass_state)
        self.assertTrue(torchbearermodel._test_loop.call_args[0][2] == pass_state)

    def test_evaluate_generator_steps(self):
        torchmodel = MagicMock()
        optimizer = MagicMock()
        generator = MagicMock()

        pass_state = False
        steps = 100

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.main_state[torchbearer.METRICS] = 1
        torchbearermodel._test_loop = Mock()

        torchbearermodel.evaluate_generator(generator, 0, steps, pass_state)
        self.assertTrue(torchbearermodel._test_loop.call_args[0][4] == steps)

    def test_predict(self):
        x = torch.rand(1,5)
        pass_state = False
        verbose=0

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()
        metric = Metric('test')

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearermodel = Model(torchmodel, optimizer, criterion, [metric])
        torchbearermodel.predict_generator = Mock()
        pred = torchbearermodel.predict_generator
        torchbearermodel.predict(x, verbose=verbose, pass_state=pass_state)

        pred.assert_called_once()
        self.assertTrue(pred.call_args[0][1] == verbose)
        self.assertTrue(pred.call_args[1]['pass_state'] == pass_state)

    def test_predict_generator_args(self):
        from torchbearer.callbacks import AggregatePredictions

        torchmodel = MagicMock()
        optimizer = MagicMock()
        generator = MagicMock()

        pass_state = False
        steps = None

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.main_state[torchbearer.FINAL_PREDICTIONS] = 1
        torchbearermodel._test_loop = Mock()

        torchbearermodel.predict_generator(generator, 0, steps, pass_state)
        self.assertIsInstance(torchbearermodel._test_loop.call_args[0][1].callback_list[0], AggregatePredictions)
        self.assertTrue(torchbearermodel._test_loop.call_args[0][2] == pass_state)
        self.assertTrue(torchbearermodel._test_loop.call_args[0][4] == steps)

    def test_predict_generator_verbose(self):
        from torchbearer.callbacks import Tqdm

        torchmodel = MagicMock()
        optimizer = MagicMock()
        generator = MagicMock()

        pass_state = False
        steps = None

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.main_state[torchbearer.FINAL_PREDICTIONS] = 1
        torchbearermodel._test_loop = Mock()

        torchbearermodel.predict_generator(generator, 1, steps, pass_state)
        self.assertIsInstance(torchbearermodel._test_loop.call_args[0][1].callback_list[1], Tqdm)
        self.assertTrue(torchbearermodel._test_loop.call_args[0][2] == pass_state)
        self.assertTrue(torchbearermodel._test_loop.call_args[0][4] == steps)

    def test_predict_generator_steps(self):
        torchmodel = MagicMock()
        optimizer = MagicMock()
        generator = MagicMock()

        pass_state = False
        steps = 100

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.main_state[torchbearer.FINAL_PREDICTIONS] = 1
        torchbearermodel._test_loop = Mock()

        torchbearermodel.predict_generator(generator, 0, steps, pass_state)
        self.assertTrue(torchbearermodel._test_loop.call_args[0][4] == steps)

    def test_predict_generator_pass_state(self):
        torchmodel = MagicMock()
        optimizer = MagicMock()
        generator = MagicMock()

        pass_state = False
        steps = 100

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.main_state[torchbearer.FINAL_PREDICTIONS] = 1
        torchbearermodel._test_loop = Mock()

        torchbearermodel.predict_generator(generator, 0, steps, pass_state)
        self.assertTrue(torchbearermodel._test_loop.call_args[0][2] == pass_state)

    def test_train(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        optimizer = MagicMock()
        metric_list = MagicMock()

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.main_state = {torchbearer.MODEL: torchmodel, torchbearer.METRIC_LIST: metric_list}
        torchbearermodel.train()
        self.assertTrue(torchbearermodel.main_state[torchbearer.MODEL].training == True)
        torchbearermodel.main_state[torchbearer.METRIC_LIST].train.assert_called_once()

    def test_eval(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        optimizer = MagicMock()
        metric_list = MagicMock()

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.main_state = {torchbearer.MODEL: torchmodel, torchbearer.METRIC_LIST: metric_list}
        torchbearermodel.eval()
        self.assertTrue(torchbearermodel.main_state[torchbearer.MODEL].training == False)
        torchbearermodel.main_state[torchbearer.METRIC_LIST].eval.assert_called_once()

    def test_to_both_args(self):
        dev = 'cuda:1'
        dtype = torch.float16

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.to = Mock()
        optimizer = torch.optim.Adam(torchmodel.parameters(), 0.1)
        state_tensor = torch.Tensor([1])
        state_tensor.to = Mock()
        optimizer.state = {'test': {'test': state_tensor}}

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.to(dev, dtype)

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

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.to(dev)

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

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.to(dtype)

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

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.to(device=dev, dtype=dtype)

        self.assertTrue(torchmodel.to.call_args[1]['device'] == dev)
        self.assertTrue(torchmodel.to.call_args[1]['dtype'] == dtype)
        self.assertTrue(state_tensor.to.call_args[1]['device'] == dev)
        self.assertTrue(state_tensor.to.call_args[1]['dtype'] == dtype)

    def test_update_device_and_dtype_from_args_only_kwarg(self):
        main_state = {}
        dtype = torch.float16
        dev = 'cuda:1'
        kwargs = {torchbearer.DEVICE: dev, torchbearer.DATA_TYPE: dtype}

        main_state = Model._update_device_and_dtype_from_args(main_state, **kwargs)

        self.assertTrue(main_state[torchbearer.DATA_TYPE] == dtype)
        self.assertTrue(main_state[torchbearer.DEVICE] == dev)

    def test_update_device_and_dtype_from_args_only_arg(self):
        main_state = {}
        dtype = torch.float16
        dev = 'cuda:1'
        args = (dtype, dev)

        main_state = Model._update_device_and_dtype_from_args(main_state, *args)

        self.assertTrue(main_state[torchbearer.DATA_TYPE] == dtype)
        self.assertTrue(main_state[torchbearer.DEVICE] == dev)

    @patch('torch.cuda.current_device')
    def test_cuda_no_device(self, device_mock):
        device_mock.return_value = 111

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.to = Mock()
        torchbearermodel.cuda()

        self.assertTrue(torchbearermodel.to.call_args[0][0] == 'cuda:' + str(111))

    def test_cuda_with_device(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.to = Mock()
        torchbearermodel.cuda(device='2')

        self.assertTrue(torchbearermodel.to.call_args[0][0] == 'cuda:2')

    def test_cpu(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.to = Mock()
        torchbearermodel.cpu()

        self.assertTrue(torchbearermodel.to.call_args[0][0] == 'cpu')

    def test_load_state_dict(self):
        key_words = {'strict': True}

        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel.load_state_dict = Mock()
        torch_state = torchmodel.state_dict()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()
        optimizer_state = optimizer.state_dict()

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearer_state = torchbearermodel.state_dict()

        torchbearermodel.load_state_dict(torchbearer_state, **key_words)

        self.assertTrue(torchmodel.load_state_dict.call_args[0][0] == torch_state)
        self.assertTrue(optimizer.load_state_dict.call_args[0][0] == optimizer_state)
        self.assertTrue(torchmodel.load_state_dict.call_args[1] == key_words)

    def test_state_dict(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        torchmodel_state = torchmodel.state_dict()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer_state = optimizer.state_dict()

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearer_state = torchbearermodel.state_dict()

        self.assertTrue(torchbearer_state[torchbearer.MODEL] == torchmodel_state)
        self.assertTrue(torchbearer_state[torchbearer.OPTIMIZER] == optimizer_state)

    def test_state_dict_kwargs(self):
        keywords = {'destination': None, 'prefix': '', 'keep_vars': False}
        torchmodel = MagicMock()
        optimizer = MagicMock()

        torchbearermodel = Model(torchmodel, optimizer, torch.nn.L1Loss(), [])
        torchbearermodel.state_dict(**keywords)

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

