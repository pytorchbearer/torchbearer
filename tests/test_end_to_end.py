import unittest

import torch
from torch.nn import Module

import torchbearer


class Net(Module):
    def __init__(self, x):
        super(Net, self).__init__()
        self.pars = torch.nn.Parameter(x)

    def f(self):
        """
        function to be minimised:
        f(x) = (x[0]-5)^2 + x[1]^2 + (x[2]-1)^2
        Solution:
        x = [5,0,1]
        """
        out = torch.zeros_like(self.pars)
        out[0] = self.pars[0]-5
        out[1] = self.pars[1]
        out[2] = self.pars[2]-1
        return torch.sum(out**2)

    def forward(self, _):
        return self.f()


class NetWithState(Net):
    def forward(self, _, state=None):
        if state is None:
            raise ValueError
        return super(NetWithState, self).forward(_)


def loss(y_pred, y_true):
    return y_pred


class TestEndToEnd(unittest.TestCase):
    def test_basic_opt(self):
        p = torch.tensor([2.0, 1.0, 10.0])
        training_steps = 1000

        model = NetWithState(p)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        tbmodel = torchbearer.Trial(model, optim, loss).for_train_steps(training_steps).for_val_steps(1)
        tbmodel.run()

        self.assertAlmostEqual(model.pars[0].item(), 5.0, places=4)
        self.assertAlmostEqual(model.pars[1].item(), 0.0, places=4)
        self.assertAlmostEqual(model.pars[2].item(), 1.0, places=4)

    def test_basic_checkpoint(self):
        p = torch.tensor([2.0, 1.0, 10.0])
        training_steps = 500

        model = Net(p)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        tbmodel = torchbearer.Trial(model, optim, loss, callbacks=[torchbearer.callbacks.MostRecent(filepath='test.pt')]).for_train_steps(training_steps).for_val_steps(1)
        tbmodel.run(2)  # Simulate 2 'epochs'

        # Reload
        p = torch.tensor([2.0, 1.0, 10.0])
        model = Net(p)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        tbmodel = torchbearer.Trial(model, optim, loss, callbacks=[torchbearer.callbacks.MostRecent(filepath='test.pt')]).for_train_steps(training_steps)
        tbmodel.load_state_dict(torch.load('test.pt'))
        self.assertEqual(len(tbmodel.state[torchbearer.HISTORY]), 2)
        self.assertAlmostEqual(model.pars[0].item(), 5.0, places=4)
        self.assertAlmostEqual(model.pars[1].item(), 0.0, places=4)
        self.assertAlmostEqual(model.pars[2].item(), 1.0, places=4)

        import os
        os.remove('test.pt')

    def test_with_loader(self):
        p = torch.tensor([2.0, 1.0, 10.0])
        training_steps = 2

        model = Net(p)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        test_var = {'loaded': False}

        def custom_loader(state):
            state[torchbearer.X], state[torchbearer.Y_TRUE] = None, None
            test_var['loaded'] = True

        tbmodel = torchbearer.Trial(model, optim, loss, callbacks=[torchbearer.callbacks.MostRecent(filepath='test.pt')]).for_train_steps(training_steps).for_val_steps(1)
        tbmodel.with_loader(custom_loader)
        self.assertTrue(not test_var['loaded'])
        tbmodel.run(1)
        self.assertTrue(test_var['loaded'])

        import os
        os.remove('test.pt')

    def test_only_model(self):
        p = torch.tensor([2.0, 1.0, 10.0])

        model = Net(p)

        tbmodel = torchbearer.Trial(model)
        self.assertListEqual(tbmodel.run(), [])

    def test_no_model(self):
        tbmodel = torchbearer.Trial(None)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            tbmodel.run()
            self.assertTrue(len(w) == 1)

        self.assertTrue(torchbearer.trial.MockModel()(torch.rand(1)) is None)

    def test_no_train_steps(self):
        tbmodel = torchbearer.Trial(None)
        tbmodel.for_val_steps(10)
        tbmodel.run()