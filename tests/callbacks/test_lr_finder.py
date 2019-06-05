from unittest import TestCase
import torchbearer
from torchbearer.callbacks import lr_finder as lrf
import numpy as np


class TestCyclicLR(TestCase):
    def test_scale_fn(self):

        finder = lrf.CyclicLR(scale_fn=None)
        self.assertTrue(finder.scale_fn(100) == 1.0)

        scaler = lambda x: x**2
        finder2 = lrf.CyclicLR(scale_fn=scaler)
        self.assertTrue(finder2.scale_fn(12) == 144)

    def test_modes(self):
        finder = lrf.CyclicLR(mode='triangular')
        self.assertTrue(finder.scale_mode == 'cycle')
        self.assertTrue(finder.scale_fn(100) == 1.0)

        true_scale_fn = lambda x: 1 / (2. ** (x - 1))
        finder2 = lrf.CyclicLR(mode='triangular2')
        self.assertTrue(finder.scale_mode == 'cycle')
        self.assertTrue(finder2.scale_fn(12) == true_scale_fn(12))

        gamma = 0.9
        true_scale_fn = lambda x: gamma**x
        finder3 = lrf.CyclicLR(mode='exp_range', gamma=gamma)
        self.assertTrue(finder3.scale_mode == 'iterations')
        self.assertTrue(finder3.scale_fn(12) == true_scale_fn(12))

    def test_next_lr_tri(self):
        finder = lrf.CyclicLR(step_size=100)
        self.assertAlmostEqual(finder.next_lr(0, 0), 0.001, places=5)
        self.assertAlmostEqual(finder.next_lr(10, 0), 0.0015000000000000005, places=5)
        self.assertAlmostEqual(finder.next_lr(100, 0), 0.006, places=5)
        self.assertAlmostEqual(finder.next_lr(110, 0), 0.006-0.0005, places=5)
        self.assertAlmostEqual(finder.next_lr(200, 0), 0.001, places=5)

    def test_next_lr_tri2(self):
        scale_fn = lambda x: 1/(2.**(x-1))
        cycle_fn = lambda x: np.floor(1 + x / (2*100))

        finder = lrf.CyclicLR(mode='triangular2', step_size=100)
        self.assertAlmostEqual(finder.next_lr(0, 0), 0.001, places=5)
        self.assertAlmostEqual(finder.next_lr(10, 0), 0.0015000000000000005, places=5)
        self.assertAlmostEqual(finder.next_lr(100, 0), 0.006, places=5)
        self.assertAlmostEqual(finder.next_lr(110, 0), (0.006-0.0005)*scale_fn(cycle_fn(110)), places=5)
        self.assertAlmostEqual(finder.next_lr(200, 0), (0.001)*scale_fn(cycle_fn(200)), places=5)

    def test_next_lr_exp_range(self):
        gamma = 0.9
        scale_fn = lambda x: gamma**x

        finder = lrf.CyclicLR(mode='exp_range', step_size=100, gamma=gamma)
        self.assertAlmostEqual(finder.next_lr(0, 0), 0.001, places=5)
        finder.iterations = 10
        self.assertAlmostEqual(finder.next_lr(10, 0), 0.0015000000000000005*scale_fn(10), places=5)
        finder.iterations = 100
        self.assertAlmostEqual(finder.next_lr(100, 0), 0.006*scale_fn(100), places=5)
        finder.iterations = 110
        self.assertAlmostEqual(finder.next_lr(110, 0), (0.006-0.0005)*scale_fn(110), places=5)
        finder.iterations = 200
        self.assertAlmostEqual(finder.next_lr(200, 0), (0.001)*scale_fn(200), places=5)

    def test_end_to_end(self):
        import torch.optim
        import torch.nn
        model = torch.nn.Linear(10, 10)
        model2 = torch.nn.Linear(10, 10)

        optim = torch.optim.SGD([
                        {'params': model.parameters()},
                        {'params': model2.parameters(), 'lr': 1e-3}
                    ], lr=1e-2, momentum=0.9)

        clr = lrf.CyclicLR(step_size=75, base_lr=[0.001, 0.0001], max_lr=[0.006, 0.0006])
        clr.on_start({torchbearer.OPTIMIZER: optim})

        lrs = []
        for i in range(100):
            clr.on_sample({torchbearer.OPTIMIZER: optim})
            clr.on_step_training({torchbearer.OPTIMIZER: optim})
            for param_group in optim.param_groups:
                lr = param_group['lr']
                lrs.append(lr)

        for i, param_group in enumerate(optim.param_groups):
            lr = param_group['lr']
            self.assertAlmostEqual(lr, 0.004399999999999999*0.1**i)

    def test_end_to_end_2(self):
        import torch.optim
        import torch.nn
        model = torch.nn.Linear(10, 10)
        model2 = torch.nn.Linear(10, 10)

        optim = torch.optim.SGD([
                        {'params': model.parameters()},
                        {'params': model2.parameters(), 'lr': 1e-3}
                    ], lr=1e-2, momentum=0.9)

        clr = lrf.CyclicLR(step_size=[75, 100], base_lr=0.001, max_lr=0.006)
        clr.on_start({torchbearer.OPTIMIZER: optim})

        lrs = []
        for i in range(100):
            clr.on_sample({torchbearer.OPTIMIZER: optim})
            clr.on_step_training({torchbearer.OPTIMIZER: optim})
            for param_group in optim.param_groups:
                lr = param_group['lr']
                lrs.append(lr)

        for i, param_group in enumerate(optim.param_groups):
            lr = param_group['lr']
            if i == 0:
                self.assertAlmostEqual(lr, 0.004399999999999999)
            if i == 1:
                self.assertAlmostEqual(lr, 0.00595)
