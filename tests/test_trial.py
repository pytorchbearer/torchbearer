from unittest import TestCase
from mock import MagicMock, Mock, patch, ANY, create_autospec

import torch
from torch.utils.data import DataLoader

import torchbearer
import torchbearer.callbacks as callbacks
from torchbearer import Trial, State
from torchbearer.metrics import Metric
from torchbearer.trial import deep_to, load_batch_none, load_batch_predict, load_batch_standard, load_batch_infinite, update_device_and_dtype, CallbackListInjection


class _StateMaker(object):
    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        assert all(isinstance(key, slice) for key in keys)
        state = State()
        for k in keys:
            state[k.start] = k.stop
        return state


make_state = _StateMaker()


class TestMockOptimizer(TestCase):
    @patch('torchbearer.trial.Optimizer')
    def test_mock_optimizer(self, mock_opt):
        mock_opt.add_param_group = Mock()
        mock_opt.load_state_dict = Mock()
        mock_opt.state_dict = Mock()
        mock_opt.step = Mock()
        mock_opt.zero_grad = Mock()

        opt = torchbearer.trial.MockOptimizer()

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

    def test_mock_optimizer_closure(self):
        t = Trial(None)
        closure = Mock()
        opt = t.state[torchbearer.OPTIMIZER]
        opt.step(closure)
        self.assertTrue(closure.call_count == 1)


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
        d = {'my_number': 10}

        @callbacks.on_start
        def set_one(state):
            d['my_number'] = 1

        set_one.on_end = Mock()

        @callbacks.on_start
        def set_two(state):
            d['my_number'] = 2

        set_two.on_end = Mock()

        injection = CallbackListInjection(set_one, callbacks.CallbackList([set_two]))

        injection.on_end({})
        self.assertEqual(set_one.on_end.call_count, 1)
        self.assertEqual(set_two.on_end.call_count, 1)

        injection.on_start({})
        self.assertEqual(d['my_number'], 2)


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

        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_GENERATOR] == generator)
        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_STEPS] == 1)

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

        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_STEPS] == 10)

    @patch('warnings.warn')
    def test_with_train_generator_inf_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_train_generator(generator, -1)

        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_STEPS] == -1)

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

        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_STEPS] == 1)

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

        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_STEPS] == -2)

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

        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_STEPS] == 2)

    @patch('warnings.warn')
    def test_with_train_generator_old_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.for_train_steps(100)
        torchbearertrial.with_train_generator(generator, None)

        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_STEPS] == 100)

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

        self.assertTrue(torchbearertrial.state[torchbearer.VALIDATION_GENERATOR] == generator)
        self.assertTrue(torchbearertrial.state[torchbearer.VALIDATION_STEPS] == 1)

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

        self.assertTrue(torchbearertrial.state[torchbearer.VALIDATION_STEPS] == 10)

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

        self.assertTrue(torchbearertrial.state[torchbearer.VALIDATION_STEPS] == 1)

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

        self.assertTrue(torchbearertrial.state[torchbearer.VALIDATION_STEPS] == -2)

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

        self.assertTrue(torchbearertrial.state[torchbearer.VALIDATION_STEPS] == 2)

    @patch('warnings.warn')
    def test_with_val_generator_old_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.for_val_steps(100)
        torchbearertrial.with_val_generator(generator, None)

        self.assertTrue(torchbearertrial.state[torchbearer.VALIDATION_STEPS] == 100)

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

        self.assertTrue(torchbearertrial.state[torchbearer.TEST_GENERATOR] == generator)
        self.assertTrue(torchbearertrial.state[torchbearer.TEST_STEPS] == 1)

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

        self.assertTrue(torchbearertrial.state[torchbearer.TEST_STEPS] == 10)

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

        self.assertTrue(torchbearertrial.state[torchbearer.TEST_STEPS] == 1)

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

        self.assertTrue(torchbearertrial.state[torchbearer.TEST_STEPS] == -2)

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

        self.assertTrue(torchbearertrial.state[torchbearer.TEST_STEPS] == 2)

    @patch('warnings.warn')
    def test_with_test_generator_old_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.for_test_steps(100)
        torchbearertrial.with_test_generator(generator, None)

        self.assertTrue(torchbearertrial.state[torchbearer.TEST_STEPS] == 100)

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

    @patch('warnings.warn')
    def test_for_inf_train_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.for_inf_train_steps()

        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_STEPS] == -1)

    @patch('warnings.warn')
    def test_for_inf_val_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.for_inf_val_steps()

        self.assertTrue(torchbearertrial.state[torchbearer.VALIDATION_STEPS] == -1)


    @patch('warnings.warn')
    def test_for_inf_test_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.for_inf_test_steps()

        self.assertTrue(torchbearertrial.state[torchbearer.TEST_STEPS] == -1)

    @patch('warnings.warn')
    def test_for_inf_steps(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.for_inf_steps(True, True, True)
        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_STEPS] == -1)
        self.assertTrue(torchbearertrial.state[torchbearer.VALIDATION_STEPS] == -1)
        self.assertTrue(torchbearertrial.state[torchbearer.TEST_STEPS] == -1)

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.for_inf_steps(True, False, True)
        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_STEPS] == -1)
        self.assertTrue(torchbearertrial.state[torchbearer.VALIDATION_STEPS] != -1)
        self.assertTrue(torchbearertrial.state[torchbearer.TEST_STEPS] == -1)

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.for_inf_steps(True, False, False)
        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_STEPS] == -1)
        self.assertTrue(torchbearertrial.state[torchbearer.VALIDATION_STEPS] != -1)
        self.assertTrue(torchbearertrial.state[torchbearer.TEST_STEPS] != -1)

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.for_inf_steps(False, False, False)
        self.assertTrue(torchbearertrial.state[torchbearer.TRAIN_STEPS] != -1)
        self.assertTrue(torchbearertrial.state[torchbearer.VALIDATION_STEPS] != -1)
        self.assertTrue(torchbearertrial.state[torchbearer.TEST_STEPS] != -1)

    @patch('warnings.warn')
    def test_with_inf_train_loader(self, _):
        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)

        optimizer = MagicMock()
        metric = Metric('test')
        criterion = None
        generator = MagicMock()
        generator.__len__.return_value = 2

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric])
        torchbearertrial.with_inf_train_loader()

        self.assertTrue(torchbearertrial.state[torchbearer.INF_TRAIN_LOADING])


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

    def test_with_data(self):
        trial = Trial(None)
        mock_train_data, mock_val_data, mock_test_data = Mock(), Mock(), Mock()
        trial.with_train_data = mock_train_data
        trial.with_val_data = mock_val_data
        trial.with_test_data = mock_test_data
        shuffle = True
        batch_size = 30

        one_tensor = torch.Tensor([1])
        target_tensor = torch.Tensor([10])
        trial.with_data(one_tensor, target_tensor, one_tensor*2, target_tensor*2, one_tensor*3, batch_size,
                        train_steps=100, val_steps=200, test_steps=300, shuffle=shuffle)

        self.assertTrue(mock_train_data.call_args[0] == (one_tensor, target_tensor, 30, shuffle, 1, 100))
        self.assertTrue(mock_val_data.call_args[0] == (one_tensor*2, target_tensor*2, 30, shuffle, 1, 200))
        self.assertTrue(mock_test_data.call_args[0] == (one_tensor*3, 30, 1, 300))


class TestWithClosureAndLoader(TestCase):
    def test_with_closure(self):
        def closure():
            return 'test'
        t = Trial(None)
        t.with_closure(closure)
        self.assertTrue(t.closure() == 'test')

    def test_with_loader(self):
        def loader(state):
            print('test')
        t = Trial(None)
        t.with_loader(loader)
        self.assertTrue(t.state[torchbearer.LOADER] == loader)


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

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback])
        torchbearertrial._fit_pass = Mock(return_value={torchbearer.METRICS: {}})
        torchbearertrial._validation_pass = Mock(return_value={torchbearer.METRICS: {}})
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

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback])
        torchbearertrial._fit_pass = Mock(return_value={torchbearer.METRICS: {}})
        torchbearertrial._validation_pass = Mock(return_value={torchbearer.METRICS: {}})
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

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback])
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

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback])
        torchbearertrial._fit_pass = Mock(return_value={torchbearer.METRICS: {}})
        torchbearertrial._validation_pass = Mock(return_value={torchbearer.METRICS: {}})
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.state[torchbearer.HISTORY] = [1,2,3,4,5]
        torchbearertrial.run(epochs=epochs, verbose=0)

        self.assertTrue(torchbearertrial._fit_pass.call_count == 5)

    @patch('warnings.warn')
    def test_run_fit_pass_Args(self, _):
        metric = Metric('test')
        metric.process = Mock(return_value={'test': 0})
        metric.process_final = Mock(return_value={'test': 0})
        metric.reset = Mock(return_value=None)

        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        train_steps = len(data)

        epochs = 1
        torchmodel = 1

        torchbearertrial = Trial(torchmodel, None, None, [], callbacks=[])
        torchbearertrial._fit_pass = Mock(return_value={torchbearer.METRICS: {}})
        torchbearertrial._validation_pass = Mock(return_value={torchbearer.METRICS: {}})
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

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback])
        torchbearertrial._fit_pass = Mock(return_value={torchbearer.METRICS: {}})
        torchbearertrial._validation_pass = Mock(return_value={torchbearer.METRICS: {}})
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        torchbearertrial.state[torchbearer.STOP_TRAINING] = True
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

        @torchbearer.callbacks.on_end_epoch
        def stop_callback(state):
            state[torchbearer.STOP_TRAINING] = True

        torchmodel = MagicMock()
        torchmodel.forward = Mock(return_value=1)
        optimizer = MagicMock()

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = Mock(return_value=loss)

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[stop_callback, callback])
        torchbearertrial._fit_pass = Mock(return_value={torchbearer.METRICS: {}})
        torchbearertrial._validation_pass = Mock(return_value={torchbearer.METRICS: {}})
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

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [metric], callbacks=[callback])
        torchbearertrial._fit_pass = Mock(return_value={torchbearer.METRICS: {'fit_test': 1}})
        torchbearertrial._validation_pass = Mock(return_value={'val_test': 2})
        torchbearertrial.with_train_generator(generator, steps=train_steps)
        history = torchbearertrial.run(epochs=epochs, verbose=0)
        self.assertDictEqual(history[0], {'train_steps': train_steps, 'validation_steps': None, 'fit_test': 1, 'val_test': 2})


class TestFitPass(TestCase):
    @patch('torchbearer.CallbackListInjection')
    def test_fit_train_called(self, mock_inj):
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
        mock_inj.return_value = callback_list

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion, torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.float,
            torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: train_steps, torchbearer.EPOCH: 0, torchbearer.INF_TRAIN_LOADING: False,
            torchbearer.BACKWARD_ARGS: {},
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list,
                                  torchbearer.TRAIN_DATA: (generator, train_steps), torchbearer.INF_TRAIN_LOADING: False, torchbearer.LOADER: None}

        torchbearertrial._fit_pass(state)
        self.assertEqual(torchbearertrial.train.call_count, 1)

    @patch('torchbearer.CallbackListInjection')
    def test_fit_metrics_reset(self, mock_inj):
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
        mock_inj.return_value = callback_list

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion, torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.float,
            torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: train_steps, torchbearer.EPOCH: 0, torchbearer.INF_TRAIN_LOADING: False,
            torchbearer.BACKWARD_ARGS: {},
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list,
                                  torchbearer.TRAIN_DATA: (generator, train_steps), torchbearer.INF_TRAIN_LOADING: False, torchbearer.LOADER: None}

        torchbearertrial._fit_pass(state)
        self.assertEqual(metric_list.reset.call_count, 1)

    @patch('torchbearer.CallbackListInjection')
    def test_fit_callback_calls(self, mock_inj):
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
        mock_inj.return_value = callback_list

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion, torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.float,
            torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: train_steps, torchbearer.EPOCH: 0, torchbearer.INF_TRAIN_LOADING: False,
            torchbearer.BACKWARD_ARGS: {}
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list,
                                  torchbearer.TRAIN_DATA: (generator, train_steps), torchbearer.INF_TRAIN_LOADING: False, torchbearer.LOADER: None}

        torchbearertrial._fit_pass(state)
        self.assertEqual(callback_list.on_start_training.call_count, 1)
        self.assertTrue(callback_list.on_sample.call_count == 3)
        self.assertTrue(callback_list.on_forward.call_count == 3)
        self.assertTrue(callback_list.on_criterion.call_count == 3)
        self.assertTrue(callback_list.on_backward.call_count == 3)
        self.assertTrue(callback_list.on_step_training.call_count == 3)
        self.assertEqual(callback_list.on_end_training.call_count, 1)

    @patch('torchbearer.CallbackListInjection')
    def test_fit_optimizer_calls(self, mock_inj):
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
        mock_inj.return_value = callback_list

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion, torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.float,
            torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: train_steps, torchbearer.EPOCH: 0, torchbearer.INF_TRAIN_LOADING: False,
            torchbearer.BACKWARD_ARGS: {}
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list,
                                  torchbearer.TRAIN_DATA: (generator, train_steps), torchbearer.INF_TRAIN_LOADING: False, torchbearer.LOADER: None}

        torchbearertrial._fit_pass(state)
        self.assertTrue(optimizer.zero_grad.call_count == 3)
        self.assertTrue(optimizer.step.call_count == 3)

    @patch('torchbearer.CallbackListInjection')
    def test_fit_forward_call_no_state(self, mock_inj):
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
        mock_inj.return_value = callback_list

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion, torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.float,
            torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: train_steps, torchbearer.EPOCH: 0, torchbearer.INF_TRAIN_LOADING: False,
            torchbearer.BACKWARD_ARGS: {}
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list,
                                  torchbearer.TRAIN_DATA: (generator, train_steps), torchbearer.INF_TRAIN_LOADING: False, torchbearer.LOADER: None}

        torchbearertrial._fit_pass(state)
        self.assertTrue(torchmodel.call_count == 3)
        self.assertTrue(torchmodel.call_args_list[0][0][0].item() == 1)

    @patch('torchbearer.CallbackListInjection')
    def test_fit_forward_call_with_state(self, mock_inj):
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
        mock_inj.return_value = callback_list

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float,
            torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: train_steps, torchbearer.EPOCH: 0,
            torchbearer.BACKWARD_ARGS: {}
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list,
                                  torchbearer.TRAIN_DATA: (generator, train_steps), torchbearer.INF_TRAIN_LOADING: False, torchbearer.LOADER: None}

        torchbearertrial._fit_pass(state)
        self.assertTrue(torchmodel.call_count == 3)
        self.assertTrue(len(torchmodel.call_args_list[0][1]) == 1)

    @patch('torchbearer.CallbackListInjection')
    def test_fit_criterion(self, mock_inj):
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
        def crit_sig(y_pred, y_true):
            return loss
        criterion = create_autospec(crit_sig)

        metric_list = MagicMock()
        callback_list = MagicMock()
        mock_inj.return_value = callback_list

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer, torchbearer.INF_TRAIN_LOADING: False,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float,
            torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: train_steps, torchbearer.EPOCH: 0,
            torchbearer.BACKWARD_ARGS: {}, torchbearer.GENERATOR: generator
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list,
                                  torchbearer.TRAIN_DATA: (generator, train_steps), torchbearer.INF_TRAIN_LOADING: False, torchbearer.LOADER: None,
                                  torchbearer.GENERATOR: generator}
        torchbearertrial._fit_pass(state)
        self.assertTrue(criterion.call_count == 3)
        self.assertTrue(criterion.call_args_list[0][0][0] == 5)
        self.assertTrue(criterion.call_args_list[0][0][1].item() == 1.0)

    def test_fit_criterion_passed_state(self):
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
        def crit_sig(state):
            return loss
        criterion = create_autospec(crit_sig)

        metric_list = MagicMock()
        callback_list = MagicMock()
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer, torchbearer.INF_TRAIN_LOADING: False, torchbearer.LOADER: None,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float,
            torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: train_steps, torchbearer.EPOCH: 0,
            torchbearer.BACKWARD_ARGS: {}
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list, torchbearer.LOADER: None,
                                  torchbearer.TRAIN_DATA: (generator, train_steps), torchbearer.INF_TRAIN_LOADING: False,}

        torchbearertrial._fit_pass(state)
        self.assertTrue(criterion.call_count == 3)
        self.assertTrue(criterion.call_args_list[0][0][0] == state)


    @patch('torchbearer.CallbackListInjection')
    def test_fit_backward(self, mock_inj):
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
        mock_inj.return_value = callback_list

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer, torchbearer.INF_TRAIN_LOADING: False,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.LOADER: None,
            torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: train_steps, torchbearer.EPOCH: 0,
            torchbearer.BACKWARD_ARGS: {}
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list,
                                  torchbearer.INF_TRAIN_LOADING: False, torchbearer.TRAIN_DATA: (generator, train_steps), torchbearer.LOADER: None}

        torchbearertrial._fit_pass(state)
        self.assertTrue(loss.backward.call_count == 3)

    @patch('torchbearer.CallbackListInjection')
    def test_fit_metrics_process(self, mock_inj):
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
        mock_inj.return_value = callback_list

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer, torchbearer.INF_TRAIN_LOADING: False, torchbearer.BACKWARD_ARGS: {},
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float,
            torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: train_steps, torchbearer.EPOCH: 0
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list,
                                  torchbearer.TRAIN_DATA: (generator, train_steps), torchbearer.INF_TRAIN_LOADING: False, torchbearer.LOADER: None}

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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer, torchbearer.INF_TRAIN_LOADING: False, torchbearer.BACKWARD_ARGS: {},
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float,
            torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: train_steps, torchbearer.EPOCH: 0
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list,
                                  torchbearer.TRAIN_DATA: (generator, train_steps), torchbearer.INF_TRAIN_LOADING: False, torchbearer.LOADER: None}

        history = torchbearertrial._fit_pass(state)[torchbearer.METRICS]
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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: True, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer, torchbearer.INF_TRAIN_LOADING: False,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.BACKWARD_ARGS: {},
            torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: train_steps, torchbearer.EPOCH: 0
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list,
                                  torchbearer.TRAIN_DATA: (generator, train_steps), torchbearer.INF_TRAIN_LOADING: False, torchbearer.LOADER: None}

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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: True, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer, torchbearer.BACKWARD_ARGS: {},
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: None, torchbearer.TRAIN_STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1]
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: None, torchbearer.CALLBACK_LIST: callback_list, torchbearer.TRAIN_DATA: (None, steps), torchbearer.LOADER: None}

        state = torchbearertrial._fit_pass(state)
        self.assertTrue(state[torchbearer.ITERATOR] is None)

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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: True, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.TRAIN_GENERATOR: generator, torchbearer.TRAIN_STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1], torchbearer.INF_TRAIN_LOADING: False,
            torchbearer.BACKWARD_ARGS: {}
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.TRAIN_GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list,
                                  torchbearer.TRAIN_DATA: (generator, steps), torchbearer.INF_TRAIN_LOADING: False, torchbearer.LOADER: None}

        state = torchbearertrial._fit_pass(state)
        self.assertTrue(state[torchbearer.ITERATOR] is not None)
        self.assertTrue(state[torchbearer.Y_PRED] == 5)
        self.assertTrue(state[torchbearer.LOSS].item() == 2)
        self.assertTrue(state[torchbearer.METRICS]['test'] == 2)


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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.GENERATOR: generator, torchbearer.STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1], torchbearer.SAMPLER: load_batch_none
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list}

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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.GENERATOR: generator, torchbearer.STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1], torchbearer.SAMPLER: load_batch_none
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list}

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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.GENERATOR: generator, torchbearer.STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1], torchbearer.SAMPLER: load_batch_standard
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list}

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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.GENERATOR: generator, torchbearer.STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1], torchbearer.SAMPLER: load_batch_none
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = True
        torchbearertrial.state = {torchbearer.GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list}

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

        def spec_crit(y_pred, y_true):
            pass

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = create_autospec(spec_crit)
        criterion.return_value = loss

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.GENERATOR: generator, torchbearer.STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1], torchbearer.SAMPLER: load_batch_standard
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list}

        torchbearertrial._test_pass(state)
        self.assertTrue(criterion.call_count == 3)
        self.assertTrue(criterion.call_args_list[0][0][0] == 5)
        self.assertTrue(criterion.call_args_list[0][0][1].item() == 1.0)

    def test_criterion_passed_state(self):
        data = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])),
                (torch.Tensor([3]), torch.Tensor([3]))]
        generator = DataLoader(data)
        steps = len(data)
        epochs = 1
        torchmodel = MagicMock()
        torchmodel.return_value = 5
        optimizer = MagicMock()

        def spec_crit(state):
            pass

        loss = torch.tensor([2.0], requires_grad=True)
        criterion = create_autospec(spec_crit)
        criterion.return_value = loss

        metric_list = MagicMock()
        metric_list.process.return_value = {'test': 0}
        metric_list.process_final.return_value = {'test': 2}
        callback_list = MagicMock()
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu', torchbearer.LOADER: None,
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.GENERATOR: generator, torchbearer.STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1], torchbearer.SAMPLER: load_batch_standard
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list, torchbearer.LOADER: None,}

        torchbearertrial._test_pass(state)
        self.assertTrue(criterion.call_count == 3)
        self.assertTrue(criterion.call_args_list[0][0][0] == state)

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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.GENERATOR: generator, torchbearer.STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1], torchbearer.SAMPLER: load_batch_none
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list}

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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: False, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.GENERATOR: generator, torchbearer.STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1], torchbearer.SAMPLER: load_batch_none
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list}

        history = torchbearertrial._test_pass(state)
        self.assertEqual(metric_list.process_final.call_count, 1)
        self.assertTrue(history[torchbearer.METRICS]['test'] == 2)

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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: True, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.GENERATOR: generator, torchbearer.STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1], torchbearer.SAMPLER: load_batch_none
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list}

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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: True, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.GENERATOR: None, torchbearer.STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1], torchbearer.SAMPLER: load_batch_none
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.GENERATOR: None, torchbearer.CALLBACK_LIST: callback_list}

        state = torchbearertrial._test_pass(state)
        self.assertTrue(state[torchbearer.ITERATOR] is None)

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
        torchbearer.CallbackListInjection = Mock(return_value=callback_list)

        state = make_state[
            torchbearer.MAX_EPOCHS: epochs, torchbearer.STOP_TRAINING: True, torchbearer.MODEL: torchmodel, torchbearer.CRITERION: criterion,
            torchbearer.OPTIMIZER: optimizer,
            torchbearer.METRIC_LIST: metric_list, torchbearer.CALLBACK_LIST: callback_list, torchbearer.DEVICE: 'cpu',
            torchbearer.DATA_TYPE: torch.float, torchbearer.HISTORY: [], torchbearer.GENERATOR: generator, torchbearer.STEPS: steps, torchbearer.EPOCH: 0,
            torchbearer.X: data[0][0], torchbearer.Y_TRUE: data[0][1], torchbearer.SAMPLER: load_batch_none
        ]

        torchbearertrial = Trial(torchmodel, optimizer, criterion, [], callbacks=[])
        torchbearertrial.train = Mock()
        torchbearertrial.pass_state = False
        torchbearertrial.state = {torchbearer.GENERATOR: generator, torchbearer.CALLBACK_LIST: callback_list}

        state = torchbearertrial._test_pass(state)
        self.assertTrue(state[torchbearer.ITERATOR] is not None)
        self.assertTrue(state[torchbearer.Y_PRED] == 5)
        self.assertTrue(state[torchbearer.LOSS].item() == 2)
        self.assertTrue(state[torchbearer.METRICS]['test'] == 2)


class TestTrialValEvalPred(TestCase):
    def test_validation_pass(self):
        generator = MagicMock()
        steps = 5
        torchbearer.CallbackListInjection = Mock()

        state = {torchbearer.VALIDATION_GENERATOR: generator, torchbearer.VALIDATION_STEPS: steps, torchbearer.METRICS: 1}
        t = Trial(MagicMock())
        eval_mock = t.eval = Mock()
        test_pass_mock = t._test_pass = Mock()
        t.state = {torchbearer.VALIDATION_GENERATOR: generator, torchbearer.CALLBACK_LIST: None,
                   torchbearer.VALIDATION_DATA: (generator, steps), torchbearer.LOADER: None}
        metrics = t._validation_pass(state)

        self.assertEqual(eval_mock.call_count, 1)
        self.assertEqual(test_pass_mock.call_count, 1)
        test_pass_state = test_pass_mock.call_args[0][0]
        self.assertTrue(test_pass_state[torchbearer.GENERATOR] == generator)
        self.assertTrue(test_pass_state[torchbearer.STEPS] == steps)
        self.assertTrue(metrics == 1)

    def test_validation_pass_none(self):
        generator = None
        steps = None
        torchbearer.CallbackListInjection = Mock()

        state = {torchbearer.VALIDATION_GENERATOR: generator, torchbearer.VALIDATION_STEPS: steps, torchbearer.METRICS: 1}
        t = Trial(MagicMock())
        eval_mock = t.eval = Mock()
        t._test_pass = Mock()
        t.state = {torchbearer.VALIDATION_GENERATOR: generator, torchbearer.CALLBACK_LIST: None,
                   torchbearer.VALIDATION_DATA: (generator, steps), torchbearer.LOADER: None}
        t._validation_pass(state)

        self.assertTrue(eval_mock.call_count == 0)

    def test_evaluate(self):
        generator = MagicMock()
        steps = 5
        torchbearer.CallbackListInjection = Mock()

        t = Trial(MagicMock())
        eval_mock = t.eval = Mock()
        clist = MagicMock()
        state = {torchbearer.HISTORY: [{'train_steps': 'steps', 'train_metric': 2}], torchbearer.VALIDATION_GENERATOR: generator,
                 torchbearer.CALLBACK_LIST: clist, torchbearer.VALIDATION_STEPS: steps, torchbearer.VALIDATION_DATA: (generator, steps),
                 torchbearer.METRICS: {'val_metric': 1}, torchbearer.LOADER: None}
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
        self.assertTrue(test_pass_state[torchbearer.GENERATOR] == generator)
        self.assertTrue(test_pass_state[torchbearer.STEPS] == steps)
        self.assertEqual(metrics['val_metric'], 1)
        self.assertDictEqual(state[torchbearer.HISTORY][0], {'train_steps': 'steps', 'train_metric': 2, 'val_metric': 1})

    def test_evaluate_none(self):
        generator = None
        steps = None
        torchbearer.CallbackListInjection = Mock()

        t = Trial(MagicMock())
        eval_mock = t.eval = Mock()
        test_pass_mock = t._test_pass = Mock(return_value={torchbearer.METRICS: 1})
        t.state = {torchbearer.VALIDATION_GENERATOR: generator, torchbearer.CALLBACK_LIST: None,
                   torchbearer.VALIDATION_STEPS: steps, torchbearer.VALIDATION_DATA: (generator, steps), torchbearer.LOADER: None}
        metrics = t.evaluate()

        self.assertTrue(eval_mock.call_count == 0)

    def test_predict(self):
        generator = MagicMock()
        steps = 5
        torchbearer.CallbackListInjection = Mock()

        state = {torchbearer.TEST_GENERATOR: generator, torchbearer.TEST_STEPS: steps, torchbearer.METRICS: 1}
        t = Trial(MagicMock())
        eval_mock = t.eval = Mock()
        test_pass_mock = t._test_pass = Mock(return_value={torchbearer.FINAL_PREDICTIONS: 1})
        clist = MagicMock()
        t.state = {torchbearer.TEST_GENERATOR: generator, torchbearer.CALLBACK_LIST: clist, torchbearer.TEST_STEPS: steps,
                   torchbearer.TEST_DATA: (generator, steps), torchbearer.LOADER: None}
        metrics = t.predict()

        self.assertEqual(clist.on_start.call_count, 1)
        self.assertEqual(clist.on_start_epoch.call_count, 1)
        self.assertEqual(clist.on_end_epoch.call_count, 1)
        self.assertEqual(clist.on_end.call_count, 1)
        self.assertEqual(eval_mock.call_count, 1)
        self.assertEqual(test_pass_mock.call_count, 1)
        test_pass_state = test_pass_mock.call_args[0][0]
        self.assertTrue(test_pass_state[torchbearer.GENERATOR] == generator)
        self.assertTrue(test_pass_state[torchbearer.STEPS] == steps)
        self.assertTrue(metrics == 1)

    def test_predict_none(self):
        generator = None
        steps = None
        torchbearer.CallbackListInjection = Mock()

        state = {torchbearer.TEST_GENERATOR: generator, torchbearer.TEST_STEPS: steps, torchbearer.METRICS: 1}
        t = Trial(MagicMock())
        eval_mock = t.eval = Mock()
        test_pass_mock = t._test_pass = Mock(return_value={torchbearer.FINAL_PREDICTIONS: 1})
        t.state = {torchbearer.TEST_GENERATOR: generator, torchbearer.CALLBACK_LIST: None, torchbearer.TEST_STEPS: steps,
                   torchbearer.TEST_DATA: (generator, steps), torchbearer.LOADER: None}
        metrics = t.predict()

        self.assertTrue(eval_mock.call_count == 0)


class TestReplay(TestCase):
    @patch('torchbearer.trial.Tqdm')
    def test_replay_tqdm(self, tq):
        t = Trial(MagicMock())
        callback = MagicMock()
        history = [{'train_steps': 10, 'validation_steps': 5, 'test': i, 'val_test2': i+1} for i in range(10)]

        t.state[torchbearer.HISTORY] = history
        t.replay(callbacks=[callback])
        self.assertEqual(tq.call_count, 1)

    @patch('torchbearer.trial.Tqdm')
    def test_replay_no_tqdm(self, tq):
        t = Trial(MagicMock())
        callback = MagicMock()
        history = [{'train_steps': 10, 'validation_steps': 5, 'test': i, 'val_test2': i+1} for i in range(10)]

        t.state[torchbearer.HISTORY] = history
        t.replay(callbacks=[callback], verbose=0)
        tq.assert_not_called()

    @patch('torchbearer.trial.Tqdm')
    def test_replay_multi_call(self, mock_tqdm):
        t = Trial(MagicMock())
        history = [{'train_steps': 10, 'validation_steps': 5, 'test': i, 'val_test2': i + 1} for i in range(1)]

        t.state[torchbearer.HISTORY] = history
        t.replay(verbose=2)
        mock_tqdm.reset_mock()
        callback = MagicMock()
        t.replay(callbacks=[callback], verbose=0)
        mock_tqdm.assert_not_called()

    def test_replay_callback_calls(self):
        t = Trial(MagicMock())
        callback = MagicMock()
        history = [{'train_steps': 10, 'validation_steps': 5, 'test': i, 'val_test2': i+1} for i in range(10)]

        t.state[torchbearer.HISTORY] = history
        t.replay(callbacks=[callback], verbose=0)
        self.assertEqual(callback.on_start.call_count, 1)
        self.assertTrue(callback.on_sample.call_count == 100)
        self.assertTrue(callback.on_sample_validation.call_count == 50)

    def test_replay_none_train_steps(self):
        t = Trial(MagicMock())
        callback = MagicMock()
        history = [{'train_steps': None, 'validation_steps': 5, 'test': i, 'val_test2': i+1} for i in range(10)]

        t.state[torchbearer.HISTORY] = history
        t.replay(callbacks=[callback], verbose=0)
        self.assertEqual(callback.on_start.call_count, 1)
        self.assertTrue(callback.on_sample.call_count == 0)
        self.assertTrue(callback.on_sample_validation.call_count == 50)

    def test_replay_none_validation_steps(self):
        t = Trial(MagicMock())
        callback = MagicMock()
        history = [{'train_steps': 10, 'validation_steps': None, 'test': i} for i in range(10)]

        t.state[torchbearer.HISTORY] = history
        t.replay(callbacks=[callback], verbose=0)
        self.assertEqual(callback.on_start.call_count, 1)
        self.assertTrue(callback.on_sample.call_count == 100)
        self.assertTrue(callback.on_sample_validation.call_count == 0)

    def test_replay_one_batch_true(self):
        t = Trial(MagicMock())
        callback = MagicMock()
        history = [{'train_steps': 10, 'validation_steps': 5, 'test': i, 'val_test2': i+1} for i in range(1)]

        t.state[torchbearer.HISTORY] = history
        t.replay(callbacks=[callback], verbose=0, one_batch=True)
        self.assertTrue(callback.on_start.call_count == 1)
        self.assertTrue(callback.on_sample.call_count == 1)
        self.assertTrue(callback.on_sample_validation.call_count == 1)

    def test_replay_metrics(self):
        t = Trial(MagicMock())
        callback = MagicMock()
        history = [{'train_steps': 10, 'validation_steps': 5, 'test': i, 'val_test2': i+1} for i in range(10)]

        t.state[torchbearer.HISTORY] = history
        t.replay(callbacks=[callback], verbose=0)

        self.assertTrue(callback.on_sample.call_args_list[0][0][0][torchbearer.METRICS]['test'] == 9)
        self.assertTrue(callback.on_sample_validation.call_args_list[0][0][0][torchbearer.METRICS]['val_test2'] == 10)

    def test_replay_stop_training(self):
        t = Trial(MagicMock())
        callback = MagicMock()

        @torchbearer.callbacks.on_sample
        def stop_training(state):
            state[torchbearer.STOP_TRAINING] = True

        history = [{'train_steps': 10, 'validation_steps': 5, 'test': i, 'val_test2': i+1} for i in range(10)]

        t.state[torchbearer.HISTORY] = history
        t.replay(callbacks=[callback, stop_training], verbose=0)

        self.assertTrue(callback.on_sample.call_count == 10)
        callback.on_sample_validation.assert_not_called()

    def test_replay_stop_training_on_validation(self):
        t = Trial(MagicMock())
        callback = MagicMock()

        @torchbearer.callbacks.on_sample_validation
        def stop_training(state):
            state[torchbearer.STOP_TRAINING] = True

        history = [{'train_steps': 10, 'validation_steps': 5, 'test': i, 'val_test2': i+1} for i in range(10)]

        t.state[torchbearer.HISTORY] = history
        t.replay(callbacks=[callback, stop_training], verbose=0)

        self.assertTrue(callback.on_sample_validation.call_count == 1)


class TestTrialMembers(TestCase):
    def test_init_none_criterion(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        optimizer = MagicMock()
        metric = MagicMock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [metric], []).to('cpu', torch.float64)
        loss = torchbearertrial.state[torchbearer.CRITERION](None, None)
        self.assertTrue(str(loss.device) == 'cpu')
        self.assertTrue(loss.dtype == torch.float64)
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(loss.shape == torch.Size([1]))
        self.assertTrue(loss.item() == 0)

    def test_init_none_criterion_add(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        optimizer = MagicMock()
        metric = MagicMock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [metric], []).to('cpu', torch.float64)
        loss = torchbearertrial.state[torchbearer.CRITERION](None, None)
        loss = loss + 1
        self.assertTrue(str(loss.device) == 'cpu')
        self.assertTrue(loss.dtype == torch.float64)
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(loss.shape == torch.Size([1]))
        self.assertTrue(loss.item() == 1)

    def test_str(self):
        torchmodel = "mod"
        optimizer = "opt"
        metric = torchbearer.metrics.Metric('met')
        cb = torchbearer.callbacks.Callback()
        cb.on_init = Mock()

        torchbearertrial = Trial(torchmodel, optimizer, "crit", [metric], [cb])
        correct_string = "--------------------- OPTIMZER ---------------------\nopt\n\n-------------------- CRITERION ---------------------\ncrit\n\n--------------------- METRICS ----------------------\n['met']\n\n-------------------- CALLBACKS ---------------------\n['torchbearer.bases.Callback']\n\n---------------------- MODEL -----------------------\nmod\n\n"
        self.assertEqual(str(torchbearertrial), correct_string)
        self.assertEqual(cb.on_init.call_count, 1)

    def test_repr(self):
        torchmodel = "mod"
        optimizer = "opt"
        metric = torchbearer.metrics.Metric('met')

        torchbearertrial = Trial(torchmodel, optimizer, "crit", [metric], [torchbearer.callbacks.Callback()])
        self.assertEqual(str(torchbearertrial), repr(torchbearertrial))

    def test_train(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        optimizer = MagicMock()
        metric = MagicMock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [metric], [])
        torchbearertrial.train()
        self.assertTrue(torchbearertrial.state[torchbearer.MODEL].training == True)
        self.assertEqual(metric.train.call_count, 1)

    def test_eval(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1,1))
        optimizer = MagicMock()
        metric = MagicMock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [metric], [])
        torchbearertrial.eval()
        self.assertTrue(torchbearertrial.state[torchbearer.MODEL].training == False)
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

        torchbearertrial = Trial(torchmodel, optimizer, None, [], [])
        torchbearertrial.state[torchbearer.CALLBACK_LIST] = callback_list
        torchbearertrial.state[torchbearer.HISTORY] = history
        torchbearer_state = torchbearertrial.state_dict()
        torchbearertrial.state[torchbearer.HISTORY] = 'Wrong'

        torchbearertrial.load_state_dict(torchbearer_state, **key_words)

        self.assertTrue(torchmodel.load_state_dict.call_args[0][0] == torch_state)
        self.assertTrue(optimizer.load_state_dict.call_args[0][0] == optimizer_state)
        self.assertTrue(optimizer.load_state_dict.call_args[0][0] == optimizer_state)
        self.assertTrue(callback_list.load_state_dict.call_args[0][0] == 1)

        self.assertTrue(torchbearertrial.state[torchbearer.HISTORY] == history)
        self.assertEqual(torchbearertrial.state[torchbearer.MODEL].load_state_dict.call_count, 1)
        self.assertEqual(torchbearertrial.state[torchbearer.OPTIMIZER].load_state_dict.call_count, 1)
        self.assertEqual(torchbearertrial.state[torchbearer.CALLBACK_LIST].load_state_dict.call_count, 1)
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

        torchbearertrial = Trial(torchmodel, optimizer, None, [], [])
        torchbearertrial.state[torchbearer.HISTORY] = history
        torchbearer_state = torchbearertrial.state_dict()
        torchbearertrial.state[torchbearer.HISTORY] = 'Wrong'

        torchbearertrial.load_state_dict(torchbearer_state, resume=False, **key_words)

        self.assertTrue(torchbearertrial.state[torchbearer.HISTORY] is 'Wrong')
        self.assertEqual(torchbearertrial.state[torchbearer.MODEL].load_state_dict.call_count, 1)
        self.assertTrue(torchbearertrial.state[torchbearer.OPTIMIZER].load_state_dict.call_count == 0)
        self.assertTrue(torchmodel.load_state_dict.call_args[1] == key_words)

    def test_load_state_dict_wrong_version(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1, 1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [], [])

        torchbearer_state = torchbearertrial.state_dict()
        torchbearer_state[torchbearer.VERSION] = '0.1.7'  # Old version

        import warnings
        with warnings.catch_warnings(record=True) as w:
            torchbearertrial.load_state_dict(torchbearer_state, resume=True)
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))

    def test_load_state_dict_not_torchbearer(self):
        torchmodel = torch.nn.Sequential(torch.nn.Linear(1, 1))
        torchmodel.load_state_dict = Mock()

        optimizer = torch.optim.SGD(torchmodel.parameters(), 0.1)
        optimizer.load_state_dict = Mock()

        torchbearertrial = Trial(torchmodel, optimizer, None, [], [])

        torchbearer_state = torchbearertrial.state_dict()
        torchbearer_state[torchbearer.VERSION] = '0.1.7'  # Old version

        import warnings
        with warnings.catch_warnings(record=True) as w:
            torchbearertrial.load_state_dict(torchbearer_state[torchbearer.MODEL])
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))

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
        torchbearertrial.state[torchbearer.HISTORY] = history
        torchbearertrial.state[torchbearer.CALLBACK_LIST] = callback_list
        torchbearer_state = torchbearertrial.state_dict()

        self.assertTrue(torchbearer_state[torchbearer.VERSION] == torchbearer.__version__.replace('.dev', ''))
        self.assertTrue(torchbearer_state[torchbearer.MODEL] == torchmodel_state)
        self.assertTrue(torchbearer_state[torchbearer.OPTIMIZER] == optimizer_state)
        self.assertTrue(torchbearer_state[torchbearer.CALLBACK_LIST] == 1)
        self.assertTrue(torchbearer_state[torchbearer.HISTORY] == history)

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

        printer = torchbearer.trial.get_printer(verbose=verbose, validation_label_letter=validation_label_letter)
        tq.assert_not_called()

    @patch('torchbearer.trial.Tqdm')
    def test_get_printer_verbose_1(self, tq):
        verbose = 1
        validation_label_letter = 'v'

        printer = torchbearer.trial.get_printer(verbose=verbose, validation_label_letter=validation_label_letter)
        tq.assert_called_once_with(on_epoch=True, validation_label_letter=validation_label_letter)

    @patch('torchbearer.trial.Tqdm')
    def test_get_printer_verbose_2(self, tq):
        verbose = 2
        validation_label_letter = 'v'

        printer = torchbearer.trial.get_printer(verbose=verbose, validation_label_letter=validation_label_letter)
        tq.assert_called_once_with(validation_label_letter=validation_label_letter)

    @patch('torchbearer.trial.Tqdm')
    def test_get_printer_letter(self, tq):
        verbose = 2
        validation_label_letter = 'r'

        printer = torchbearer.trial.get_printer(verbose=verbose, validation_label_letter=validation_label_letter)
        tq.assert_called_once_with(validation_label_letter=validation_label_letter)

    @patch('torchbearer.trial.get_printer')
    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_printer_no_tqdm(self, c_inj, get_print_mock):
        callback_list = torchbearer.callbacks.CallbackList([])

        class SomeClass:
            @torchbearer.inject_printer('v')
            def test_func(self, verbose=0):
                pass

        t = SomeClass()
        t.state = {torchbearer.CALLBACK_LIST: callback_list}
        t.test_func(verbose=0)
        self.assertEqual(c_inj.call_count, 1)
        get_print_mock.assert_called_once_with(validation_label_letter='v', verbose=0)

    @patch('torchbearer.trial.get_printer')
    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_printer_no_kwargs(self, c_inj, get_print_mock):
        callback_list = torchbearer.callbacks.CallbackList([])

        class SomeClass:
            @torchbearer.inject_printer('v')
            def test_func(self, verbose=0):
                pass

        t = SomeClass()
        t.state = {torchbearer.CALLBACK_LIST: callback_list}
        t.test_func(1)
        self.assertEqual(c_inj.call_count, 1)
        get_print_mock.assert_called_once_with(validation_label_letter='v', verbose=1)

    @patch('torchbearer.trial.get_printer')
    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_both(self, c_inj, get_print_mock):
        callback_list = torchbearer.callbacks.CallbackList([])
        generator = MagicMock()
        steps = None

        class SomeClass:
            @torchbearer.inject_printer('v')
            @torchbearer.inject_sampler(torchbearer.GENERATOR, load_batch_standard)
            def test_func(self, verbose=0):
                pass

        t = SomeClass()
        t.state = {torchbearer.CALLBACK_LIST: callback_list, torchbearer.GENERATOR: (generator, steps), torchbearer.LOADER: None}
        t.test_func(1)
        self.assertEqual(c_inj.call_count, 1)
        get_print_mock.assert_called_once_with(validation_label_letter='v', verbose=1)

    @patch('torchbearer.trial.get_printer')
    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_printer_tqdm_on_epoch(self, c_inj, get_print_mock):
        callback_list = torchbearer.callbacks.CallbackList([])

        class SomeClass:
            @torchbearer.inject_printer('t')
            def test_func(self, verbose=0):
                pass

        t = SomeClass()
        t.state = {torchbearer.CALLBACK_LIST: callback_list}
        t.test_func(verbose=1)
        self.assertEqual(c_inj.call_count, 1)
        get_print_mock.assert_called_once_with(validation_label_letter='t', verbose=1)

    @patch('torchbearer.trial.get_printer')
    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_printer_tqdm_on_batch(self, c_inj, get_print_mock):
        callback_list = torchbearer.callbacks.CallbackList([])

        class SomeClass:
            @torchbearer.inject_printer('t')
            def test_func(self, verbose=0):
                pass

        t = SomeClass()
        t.state = {torchbearer.CALLBACK_LIST: callback_list}
        t.test_func(verbose=2)
        self.assertEqual(c_inj.call_count, 1)
        get_print_mock.assert_called_once_with(validation_label_letter='t', verbose=2)

    @patch('torchbearer.trial.get_printer')
    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_printer_tqdm_default(self, c_inj, get_print_mock):
        callback_list = torchbearer.callbacks.CallbackList([])

        class SomeClass:
            @torchbearer.inject_printer('t')
            def test_func(self, verbose=2):
                pass

        t = SomeClass()
        t.state = {torchbearer.CALLBACK_LIST: callback_list}
        t.test_func()
        self.assertEqual(c_inj.call_count, 1)
        get_print_mock.assert_called_once_with(validation_label_letter='t', verbose=2)

    @patch('torchbearer.trial.Tqdm')
    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_printer_injection(self, c_inj, tq):
        callback_list = torchbearer.callbacks.CallbackList([])

        class SomeClass:
            @torchbearer.inject_printer('v')
            def test_func(self_inner, verbose=0):
                self.assertEqual(c_inj.call_count, 1)

        t = SomeClass()
        t.state = {torchbearer.CALLBACK_LIST: callback_list}
        t.test_func()
        self.assertTrue(t.state[torchbearer.CALLBACK_LIST] == callback_list)

    def test_inject_sampler_standard(self):
        generator = MagicMock()
        steps = None

        class SomeClass:
            @torchbearer.inject_sampler(torchbearer.GENERATOR, load_batch_standard)
            def test_func(self):
                pass

        t = SomeClass()
        t.state = {torchbearer.GENERATOR: (generator, steps), torchbearer.LOADER: None}
        t.test_func()
        self.assertTrue(t.state[torchbearer.SAMPLER] == torchbearer.trial.load_batch_standard)

    def test_inject_sampler_none(self):
        generator = None
        steps = None

        class SomeClass:
            @torchbearer.inject_sampler(torchbearer.GENERATOR, load_batch_standard)
            def test_func(self):
                pass

        t = SomeClass()
        t.state = {torchbearer.GENERATOR: (generator, steps), torchbearer.LOADER: None}
        t.test_func()
        self.assertTrue(t.state[torchbearer.SAMPLER] == torchbearer.trial.load_batch_none)

    def test_inject_sampler_predict(self):
        generator = MagicMock()
        steps = None

        class SomeClass:
            @torchbearer.inject_sampler(torchbearer.GENERATOR, load_batch_predict)
            def test_func(self):
                pass

        t = SomeClass()
        t.state = {torchbearer.GENERATOR: (generator, steps), torchbearer.LOADER: None}
        t.test_func()
        self.assertTrue(t.state[torchbearer.SAMPLER] == torchbearer.trial.load_batch_predict)

    def test_inject_sampler_custom(self):
        generator = MagicMock()
        steps = None

        class SomeClass:
            @torchbearer.inject_sampler(torchbearer.GENERATOR, load_batch_predict)
            def test_func(self):
                pass

        def some_loader(state):
            return 'test'

        t = SomeClass()
        t.state = {torchbearer.GENERATOR: (generator, steps), torchbearer.LOADER: some_loader}
        t.test_func()
        self.assertTrue(t.state[torchbearer.SAMPLER] == some_loader)

    @patch('warnings.warn')
    @patch('torchbearer.trial.load_batch_infinite')
    def test_inject_sampler_infinite(self, mock_lbi, _):
        generator = MagicMock()
        steps = -1

        class SomeClass:
            @torchbearer.inject_sampler(torchbearer.GENERATOR, load_batch_predict)
            def test_func(self):
                pass

        t = SomeClass()
        t.state = {torchbearer.GENERATOR: (generator, steps), torchbearer.LOADER: None}
        t.test_func()
        self.assertTrue(mock_lbi.call_args[0][0] == load_batch_predict)

    @patch('torchbearer.trial.load_batch_infinite')
    def test_inject_sampler_infinite_standard_loader(self, mock_lbi):
        class EmptyObj: # Mocks don't play well with hasattr so need an empty object
            def __len__(self):
                return 100

            def __iter__(self):
                return self

            def __next__(self):
                return None

        generator = EmptyObj()
        steps = 10

        class SomeClass:
            @torchbearer.inject_sampler(torchbearer.TRAIN_DATA, load_batch_standard)
            def test_func(self):
                pass

        t = SomeClass()
        t.state = {torchbearer.TRAIN_DATA: (generator, steps), torchbearer.INF_TRAIN_LOADING: True, torchbearer.LOADER: None}
        t.test_func()
        self.assertTrue(mock_lbi.call_args[0][0] == load_batch_standard)
        self.assertTrue(generator.tb_iter)

    @patch('torchbearer.trial.load_batch_infinite')
    def test_inject_sampler_infinite_train_loading(self, mock_lbi):
        generator = MagicMock()
        generator.__len__.return_value = 10
        steps = 5

        class SomeClass:
            @torchbearer.inject_sampler(torchbearer.TRAIN_DATA, load_batch_standard)
            def test_func(self):
                pass

        t = SomeClass()
        t.state = {torchbearer.TRAIN_DATA: (generator, steps), torchbearer.INF_TRAIN_LOADING: True, torchbearer.LOADER: None}
        t.test_func()
        self.assertTrue(mock_lbi.call_args[0][0] == load_batch_standard)

    def test_inject_sampler_data_key(self):
        generator = MagicMock()
        test_generator = 'test'
        test_steps = 1

        class SomeClass:
            @torchbearer.inject_sampler(torchbearer.GENERATOR, load_batch_predict)
            def test_func(self, data_key=None):
                pass

        t = SomeClass()
        t.state = {torchbearer.GENERATOR: (generator, None), torchbearer.TEST_GENERATOR: (test_generator, test_steps), torchbearer.LOADER: None}
        t.test_func(data_key=torchbearer.TEST_GENERATOR)
        self.assertTrue(t.state[torchbearer.GENERATOR] == test_generator)
        self.assertTrue(t.state[torchbearer.STEPS] == test_steps)

    def test_inject_sampler_data_key_no_kwargs(self):
        generator = MagicMock()
        test_generator = 'test'
        test_steps = 1

        class SomeClass:
            @torchbearer.inject_sampler(torchbearer.GENERATOR, load_batch_predict)
            def test_func(self, data_key=None):
                pass

        t = SomeClass()
        t.state = {torchbearer.GENERATOR: (generator, None), torchbearer.TEST_GENERATOR: (test_generator, test_steps), torchbearer.LOADER: None}
        t.test_func(torchbearer.TEST_GENERATOR)
        self.assertTrue(t.state[torchbearer.GENERATOR] == test_generator)
        self.assertTrue(t.state[torchbearer.STEPS] == test_steps)

    @patch('torchbearer.trial.CallbackListInjection')
    def test_inject_callback(self, c_inj):
        callback_list = torchbearer.callbacks.CallbackList([])
        test_callback = MagicMock()

        class SomeClass:
            @torchbearer.inject_callback(test_callback)
            def test_func(self_inner):
                self.assertEqual(c_inj.call_count, 1)

        t = SomeClass()
        t.state = {torchbearer.CALLBACK_LIST: callback_list}
        t.test_func()
        self.assertTrue(c_inj.call_args[0][0] == test_callback)

    def test_deep_to_tensor(self):
        base_tensor = torch.Tensor([1])
        tensor = MagicMock(spec=base_tensor)
        new_dtype = torch.float16
        new_device = 'cuda:1'

        deep_to(tensor, new_device, new_dtype)
        self.assertTrue(tensor.to.call_args[0][0] == new_device)
        self.assertTrue(tensor.to.call_args[0][1] == new_dtype)

    def test_deep_to_tensor_int_dtype(self):
        base_tensor = torch.Tensor([1])
        tensor = MagicMock(spec=base_tensor)
        tensor.dtype = torch.uint8
        new_device = 'cuda:1'
        new_dtype = torch.uint8

        deep_to(tensor, new_device, new_dtype)
        self.assertTrue(tensor.to.call_args[0][0] == new_device)
        self.assertTrue(len(tensor.to.call_args[0]) == 1)

    def test_deep_to_list(self):
        base_tensor = torch.Tensor([1])
        tensor_1 = MagicMock(spec=base_tensor)
        tensor_2 = MagicMock(spec=base_tensor)
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

    def test_deep_to_unknown_object(self):
        tensor_1 = MagicMock()
        tensor_2 = MagicMock()
        tensors = {'t1': tensor_1, 't2': tensor_2}
        new_dtype = torch.float16
        new_device = 'cuda:1'

        deep_to(tensors, new_device, new_dtype)
        self.assertTrue(tensor_1.to.call_args is None)
        self.assertTrue(tensor_2.to.call_args is None)

    def test_load_batch_standard(self):
        items = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2]))]
        iterator = iter(items)

        state = {torchbearer.ITERATOR: iterator, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.int}

        load_batch_standard(state)
        self.assertTrue(state[torchbearer.X].item() == items[0][0].item())
        self.assertTrue(state[torchbearer.Y_TRUE].item() == items[0][1].item())

    def test_load_batch_inf_standard_normal(self):
        items = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        iterator = iter(items)
        state = {torchbearer.ITERATOR: iterator, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.int}

        loader = load_batch_infinite(load_batch_standard)

        for i in range(2):
            loader(state)
        self.assertTrue(state[torchbearer.X].item() == items[1][0].item())
        self.assertTrue(state[torchbearer.Y_TRUE].item() == items[1][1].item())

    def test_load_batch_inf_standard_too_many(self):
        items = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2])), (torch.Tensor([3]), torch.Tensor([3]))]
        iterator = iter(items)

        state = {torchbearer.GENERATOR: items, torchbearer.ITERATOR: iterator, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.int}

        loader = load_batch_infinite(load_batch_standard)

        for i in range(12):
            loader(state)

        self.assertTrue(state[torchbearer.X].item() == items[2][0].item())
        self.assertTrue(state[torchbearer.Y_TRUE].item() == items[2][1].item())

    def test_load_batch_none(self):
        items = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2]))]
        iterator = iter(items)

        state = {torchbearer.ITERATOR: iterator, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.int}

        load_batch_none(state)
        self.assertTrue(state[torchbearer.X] is None)
        self.assertTrue(state[torchbearer.Y_TRUE] is None)

    def test_load_batch_predict_data(self):
        items = [torch.Tensor([1]), torch.Tensor([2])]
        iterator = iter(items)

        state = {torchbearer.ITERATOR: iterator, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.int}
        load_batch_predict(state)
        self.assertTrue(state[torchbearer.X].item() == items[0].item())

    def test_load_batch_predict_list(self):
        items = [(torch.Tensor([1]), torch.Tensor([1])), (torch.Tensor([2]), torch.Tensor([2]))]
        iterator = iter(items)

        state = {torchbearer.ITERATOR: iterator, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.int}

        load_batch_predict(state)
        self.assertTrue(state[torchbearer.X].item() == items[0][0].item())
        self.assertTrue(state[torchbearer.Y_TRUE].item() == items[0][1].item())

    def test_update_device_and_dtype_only_kwarg(self):
        main_state = {}
        dtype = torch.float16
        dev = 'cuda:1'

        kwargs = {str(torchbearer.DEVICE): dev, str(torchbearer.DATA_TYPE): dtype}

        main_state = update_device_and_dtype(main_state, **kwargs)

        self.assertTrue(main_state[torchbearer.DATA_TYPE] == dtype)
        self.assertTrue(main_state[torchbearer.DEVICE] == dev)

    def test_update_device_and_dtype_only_arg(self):
        main_state = {}
        dtype = torch.float16
        dev = 'cuda:1'
        args = (dtype, dev)

        main_state = update_device_and_dtype(main_state, *args)

        self.assertTrue(main_state[torchbearer.DATA_TYPE] == dtype)
        self.assertTrue(main_state[torchbearer.DEVICE] == dev)

    def test_new_iter_none(self):
        generator = None
        t = Trial(None)
        out = t._new_iter(generator)
        self.assertTrue(out is None)

    def test_new_iter_standard(self):
        class EmptyObj(object):
            def __init__(self):
                super(self.__class__, self).__init__()
                self.count = 0

            def __iter__(self):
                self.count += 1
                return iter([1,2,3])

        generator = EmptyObj()
        t = Trial(None)
        _ = t._new_iter(generator)
        self.assertTrue(generator.count == 1)
        self.assertTrue(not hasattr(generator, 'inf'))

    def test_new_iter_inf(self):
        class EmptyObj(object):
            def __init__(self):
                super(self.__class__, self).__init__()
                self.count = 0
                self.tb_iter = Mock()
                self.inf = True

            def __iter__(self):
                self.count += 1
                return iter([1,2,3])

        generator = EmptyObj()
        t = Trial(None)
        out = t._new_iter(generator)
        self.assertTrue(out == generator.tb_iter)
        self.assertTrue(generator.count == 0)
