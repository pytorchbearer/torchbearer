from unittest import TestCase
from mock import patch, Mock
import warnings

import torchbearer
from torchbearer.bases import _pytorch_version_gt
from torchbearer.callbacks import TorchScheduler, LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,\
    ReduceLROnPlateau, CyclicLR


class TestTorchScheduler(TestCase):
    def setUp(self):
        super(TestTorchScheduler, self).setUp()
        warnings.filterwarnings('always')

    def tearDown(self):
        super(TestTorchScheduler, self).tearDown()
        warnings.filterwarnings('default')

    def test_torch_scheduler_on_batch_with_monitor(self):
        state = {torchbearer.EPOCH: 1, torchbearer.METRICS: {'test': 101}, torchbearer.OPTIMIZER: 'optimizer', torchbearer.MODEL: Mock()}
        mock_scheduler = Mock()
        mock_scheduler.return_value = mock_scheduler

        torch_scheduler = TorchScheduler(mock_scheduler, monitor='test', step_on_batch=True)
        torch_scheduler._newstyle = True

        import warnings
        with warnings.catch_warnings(record=True) as w:
            torch_scheduler.on_start(state)
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
        mock_scheduler.assert_called_once_with('optimizer', last_epoch=0)
        mock_scheduler.reset_mock()

        torch_scheduler.on_start_training(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_sample(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_step_training(state)
        mock_scheduler.step.assert_called_once_with(101)
        mock_scheduler.reset_mock()

        torch_scheduler.on_end_epoch(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

    def test_torch_scheduler_on_batch_with_monitor_oldstyle(self):
        state = {torchbearer.EPOCH: 1, torchbearer.METRICS: {'test': 101}, torchbearer.OPTIMIZER: 'optimizer', torchbearer.MODEL: Mock()}
        mock_scheduler = Mock()
        mock_scheduler.return_value = mock_scheduler

        torch_scheduler = TorchScheduler(mock_scheduler, monitor='test', step_on_batch=True)
        torch_scheduler._newstyle = False

        import warnings
        with warnings.catch_warnings(record=True) as w:
            torch_scheduler.on_start(state)
            self.assertTrue(len(w) == 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
        mock_scheduler.assert_called_once_with('optimizer', last_epoch=0)
        mock_scheduler.reset_mock()

        torch_scheduler.on_start_training(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_sample(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_step_training(state)
        mock_scheduler.step.assert_called_once_with(101)
        mock_scheduler.reset_mock()

        torch_scheduler.on_end_epoch(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

    def test_torch_scheduler_on_epoch_with_monitor(self):
        state = {torchbearer.EPOCH: 1, torchbearer.METRICS: {'test': 101}, torchbearer.OPTIMIZER: 'optimizer',
                 torchbearer.DATA: None, torchbearer.MODEL: Mock()}
        mock_scheduler = Mock()
        mock_scheduler.return_value = mock_scheduler

        torch_scheduler = TorchScheduler(mock_scheduler, monitor='test', step_on_batch=False)
        torch_scheduler._newstyle = True

        torch_scheduler.on_start(state)
        mock_scheduler.assert_called_once_with('optimizer', last_epoch=0)
        mock_scheduler.reset_mock()

        torch_scheduler.on_start_training(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_sample(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_step_training(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_end_epoch(state)
        mock_scheduler.step.assert_called_once_with(101)
        mock_scheduler.reset_mock()

    def test_torch_scheduler_on_epoch_with_monitor_oldstyle(self):
        state = {torchbearer.EPOCH: 1, torchbearer.METRICS: {'test': 101}, torchbearer.OPTIMIZER: 'optimizer',
                 torchbearer.DATA: None, torchbearer.MODEL: Mock()}
        mock_scheduler = Mock()
        mock_scheduler.return_value = mock_scheduler

        torch_scheduler = TorchScheduler(mock_scheduler, monitor='test', step_on_batch=False)
        torch_scheduler._newstyle = False

        torch_scheduler.on_start(state)
        mock_scheduler.assert_called_once_with('optimizer', last_epoch=0)
        mock_scheduler.reset_mock()

        torch_scheduler.on_start_training(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_sample(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_step_training(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_end_epoch(state)
        mock_scheduler.step.assert_called_once_with(101, epoch=1)
        mock_scheduler.reset_mock()

    def test_torch_scheduler_on_batch_no_monitor(self):
        state = {torchbearer.EPOCH: 1, torchbearer.OPTIMIZER: 'optimizer', torchbearer.MODEL: Mock()}
        mock_scheduler = Mock()
        mock_scheduler.return_value = mock_scheduler

        torch_scheduler = TorchScheduler(mock_scheduler, monitor=None, step_on_batch=True)
        torch_scheduler._newstyle = True

        torch_scheduler.on_start(state)
        mock_scheduler.assert_called_once_with('optimizer', last_epoch=0)
        mock_scheduler.reset_mock()

        torch_scheduler.on_start_training(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_sample(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_step_training(state)
        mock_scheduler.step.assert_called_once_with()
        mock_scheduler.reset_mock()

        torch_scheduler.on_end_epoch(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

    def test_torch_scheduler_on_batch_no_monitor_oldstyle(self):
        state = {torchbearer.EPOCH: 1, torchbearer.OPTIMIZER: 'optimizer', torchbearer.MODEL: Mock()}
        mock_scheduler = Mock()
        mock_scheduler.return_value = mock_scheduler

        torch_scheduler = TorchScheduler(mock_scheduler, monitor=None, step_on_batch=True)
        torch_scheduler._newstyle = False

        torch_scheduler.on_start(state)
        mock_scheduler.assert_called_once_with('optimizer', last_epoch=0)
        mock_scheduler.reset_mock()

        torch_scheduler.on_start_training(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_sample(state)
        mock_scheduler.step.assert_called_once()
        mock_scheduler.reset_mock()

        torch_scheduler.on_step_training(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_end_epoch(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

    def test_torch_scheduler_on_epoch_no_monitor(self):
        state = {torchbearer.EPOCH: 1, torchbearer.OPTIMIZER: 'optimizer', torchbearer.METRICS: {}, torchbearer.MODEL: Mock()}
        mock_scheduler = Mock()
        mock_scheduler.return_value = mock_scheduler

        torch_scheduler = TorchScheduler(mock_scheduler, monitor=None, step_on_batch=False)
        torch_scheduler._newstyle = True

        torch_scheduler.on_start(state)
        mock_scheduler.assert_called_once_with('optimizer', last_epoch=0)
        mock_scheduler.reset_mock()

        torch_scheduler.on_sample(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_step_training(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_end_epoch(state)
        mock_scheduler.step.assert_called_once()
        mock_scheduler.reset_mock()

    def test_torch_scheduler_on_epoch_no_monitor_oldstyle(self):
        state = {torchbearer.EPOCH: 1, torchbearer.OPTIMIZER: 'optimizer', torchbearer.METRICS: {}, torchbearer.MODEL: Mock()}
        mock_scheduler = Mock()
        mock_scheduler.return_value = mock_scheduler

        torch_scheduler = TorchScheduler(mock_scheduler, monitor=None, step_on_batch=False)
        torch_scheduler._newstyle = False

        torch_scheduler.on_start(state)
        mock_scheduler.assert_called_once_with('optimizer', last_epoch=0)
        mock_scheduler.reset_mock()

        torch_scheduler.on_start_training(state)
        mock_scheduler.step.assert_called_once()
        mock_scheduler.reset_mock()

        torch_scheduler.on_sample(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_step_training(state)
        mock_scheduler.step.assert_not_called()
        mock_scheduler.reset_mock()

        torch_scheduler.on_end_epoch(state)
        mock_scheduler.step.assert_called_once()
        mock_scheduler.reset_mock()        

    def test_monitor_not_found(self):
        state = {torchbearer.EPOCH: 1, torchbearer.OPTIMIZER: 'optimizer', torchbearer.METRICS: {'not_test': 1.}, torchbearer.MODEL: Mock()}
        mock_scheduler = Mock()
        mock_scheduler.return_value = mock_scheduler

        torch_scheduler = TorchScheduler(mock_scheduler, monitor='test', step_on_batch=False)
        torch_scheduler.on_start(state)

        with warnings.catch_warnings(record=True) as w:
            torch_scheduler.on_start_validation(state)
            self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            torch_scheduler.on_end_epoch(state)
            self.assertTrue('Failed to retrieve key `test`' in str(w[0].message))

    def test_monitor_found(self):
        state = {torchbearer.EPOCH: 1, torchbearer.OPTIMIZER: 'optimizer', torchbearer.METRICS: {'test': 1.}, torchbearer.MODEL: Mock()}
        mock_scheduler = Mock()
        mock_scheduler.return_value = mock_scheduler

        torch_scheduler = TorchScheduler(mock_scheduler, monitor='test', step_on_batch=False)
        torch_scheduler.on_start(state)
        with warnings.catch_warnings(record=True) as w:
            torch_scheduler.on_start_training(state)
            self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            torch_scheduler.on_start_validation(state)
            self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            torch_scheduler.on_end_epoch(state)
            self.assertTrue(len(w) == 0)

    def test_batch_monitor_not_found(self):
        state = {torchbearer.EPOCH: 1, torchbearer.OPTIMIZER: 'optimizer', torchbearer.METRICS: {'not_test': 1.}, torchbearer.MODEL: Mock()}
        mock_scheduler = Mock()
        mock_scheduler.return_value = mock_scheduler

        torch_scheduler = TorchScheduler(mock_scheduler, monitor='test', step_on_batch=True)
        torch_scheduler.on_start(state)

        with warnings.catch_warnings(record=True) as w:
            torch_scheduler.on_step_training(state)
            self.assertTrue('Failed to retrieve key `test`' in str(w[0].message))

    def test_batch_monitor_found(self):
        state = {torchbearer.EPOCH: 1, torchbearer.OPTIMIZER: 'optimizer', torchbearer.METRICS: {'test': 1.}, torchbearer.MODEL: Mock()}
        mock_scheduler = Mock()
        mock_scheduler.return_value = mock_scheduler

        torch_scheduler = TorchScheduler(mock_scheduler, monitor='test', step_on_batch=True)
        torch_scheduler.on_start(state)

        with warnings.catch_warnings(record=True) as w:
            torch_scheduler.on_step_training(state)
            self.assertTrue(len(w) == 0)


class TestLambdaLR(TestCase):
    @patch('torch.optim.lr_scheduler.LambdaLR')
    def test_lambda_lr(self, lr_mock):
        state = {torchbearer.OPTIMIZER: 'optimizer', torchbearer.EPOCH: 0, torchbearer.MODEL: Mock()}

        scheduler = LambdaLR(lr_lambda=0.1, step_on_batch=True)
        scheduler.on_start(state)

        lr_mock.assert_called_once_with('optimizer', lr_lambda=0.1, last_epoch=-1)
        self.assertTrue(scheduler._step_on_batch)


class TestStepLR(TestCase):
    @patch('torch.optim.lr_scheduler.StepLR')
    def test_lambda_lr(self, lr_mock):
        state = {torchbearer.OPTIMIZER: 'optimizer', torchbearer.EPOCH: 0, torchbearer.MODEL: Mock()}

        scheduler = StepLR(step_size=10, gamma=0.4, step_on_batch=True)
        scheduler.on_start(state)

        lr_mock.assert_called_once_with('optimizer', step_size=10, gamma=0.4, last_epoch=-1)
        self.assertTrue(scheduler._step_on_batch)


class TestMultiStepLR(TestCase):
    @patch('torch.optim.lr_scheduler.MultiStepLR')
    def test_lambda_lr(self, lr_mock):
        state = {torchbearer.OPTIMIZER: 'optimizer', torchbearer.EPOCH: 0, torchbearer.MODEL: Mock()}

        scheduler = MultiStepLR(milestones=10, gamma=0.4, step_on_batch=True)
        scheduler.on_start(state)

        lr_mock.assert_called_once_with('optimizer', milestones=10, gamma=0.4, last_epoch=-1)
        self.assertTrue(scheduler._step_on_batch)


class TestExponentialLR(TestCase):
    @patch('torch.optim.lr_scheduler.ExponentialLR')
    def test_lambda_lr(self, lr_mock):
        state = {torchbearer.OPTIMIZER: 'optimizer', torchbearer.EPOCH: 0, torchbearer.MODEL: Mock()}

        scheduler = ExponentialLR(gamma=0.4, step_on_batch=True)
        scheduler.on_start(state)

        lr_mock.assert_called_once_with('optimizer', gamma=0.4, last_epoch=-1)
        self.assertTrue(scheduler._step_on_batch)


class TestCosineAnnealingLR(TestCase):
    @patch('torch.optim.lr_scheduler.CosineAnnealingLR')
    def test_lambda_lr(self, lr_mock):
        state = {torchbearer.OPTIMIZER: 'optimizer', torchbearer.EPOCH: 0, torchbearer.MODEL: Mock()}

        scheduler = CosineAnnealingLR(T_max=4, eta_min=10, step_on_batch=True)
        scheduler.on_start(state)

        lr_mock.assert_called_once_with('optimizer', T_max=4, eta_min=10, last_epoch=-1)
        self.assertTrue(scheduler._step_on_batch)


class TestReduceLROnPlateau(TestCase):
    @patch('torch.optim.lr_scheduler.ReduceLROnPlateau')
    def test_lambda_lr(self, lr_mock):
        state = {torchbearer.OPTIMIZER: 'optimizer', torchbearer.EPOCH: 0, torchbearer.MODEL: Mock()}

        scheduler = ReduceLROnPlateau(monitor='test', mode='max', factor=0.2, patience=100, verbose=True, threshold=10,
                                      threshold_mode='thresh', cooldown=5, min_lr=0.1, eps=1e-4, step_on_batch=True)
        scheduler.on_start(state)

        lr_mock.assert_called_with('optimizer', mode='max', factor=0.2, patience=100, verbose=True, threshold=10,
                                   threshold_mode='thresh', cooldown=5, min_lr=0.1, eps=1e-4, last_epoch=-1)
        self.assertTrue(scheduler._step_on_batch)
        self.assertTrue(scheduler._monitor == 'test')


class TestCyclicLR(TestCase):
    def test_lambda_lr(self):
        if _pytorch_version_gt("1.0.0"):  # CyclicLR is implemented
            with patch('torch.optim.lr_scheduler.CyclicLR') as lr_mock:
                state = {torchbearer.OPTIMIZER: 'optimizer', torchbearer.EPOCH: 0, torchbearer.MODEL: Mock()}

                scheduler = CyclicLR(base_lr=0.01, max_lr=0.1, monitor='test', step_size_up=200, step_size_down=None,
                                     mode='triangular', gamma=2., scale_fn=None, scale_mode='cycle',
                                     cycle_momentum=False, base_momentum=0.7, max_momentum=0.9, step_on_batch=True)
                scheduler.on_start(state)

                lr_mock.assert_called_once_with('optimizer', base_lr=0.01, max_lr=0.1, step_size_up=200,
                                                step_size_down=None, mode='triangular', gamma=2., scale_fn=None,
                                                scale_mode='cycle', cycle_momentum=False, base_momentum=0.7,
                                                max_momentum=0.9, last_epoch=-1)
                self.assertTrue(scheduler._step_on_batch)
                self.assertTrue(scheduler._monitor == 'test')
        else:
            self.assertRaises(NotImplementedError,
                              lambda: CyclicLR(base_lr=0.01, max_lr=0.1, monitor='test', step_size_up=200,
                                               step_size_down=None, mode='triangular', gamma=2., scale_fn=None,
                                               scale_mode='cycle', cycle_momentum=False, base_momentum=0.7,
                                               max_momentum=0.9, step_on_batch=True))
