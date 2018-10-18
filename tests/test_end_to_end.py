import unittest

import torch
from torch.nn import Module

import torchbearer as tb


class Net(Module):
    def __init__(self, x):
        super().__init__()
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

    def forward(self, _, state):
        return self.f()


def loss(y_pred, y_true):
    return y_pred


class TestEndToEnd(unittest.TestCase):
    def test_basic_opt(self):
        p = torch.tensor([2.0, 1.0, 10.0])
        training_steps = 1000

        model = Net(p)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        tbmodel = tb.Trial(model, optim, loss, pass_state=True).for_train_steps(training_steps)
        tbmodel.run()

        self.assertAlmostEqual(model.pars[0].item(), 5.0, places=4)
        self.assertAlmostEqual(model.pars[1].item(), 0.0, places=4)
        self.assertAlmostEqual(model.pars[2].item(), 1.0, places=4)

    def test_basic_checkpoint(self):
        p = torch.tensor([2.0, 1.0, 10.0])
        training_steps = 500

        model = Net(p)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        tbmodel = tb.Trial(model, optim, loss, callbacks=[tb.callbacks.MostRecent(filepath='test.pt')],
                           pass_state=True).for_train_steps(training_steps)
        tbmodel.run(2)  # Simulate 2 'epochs'

        # Reload
        p = torch.tensor([2.0, 1.0, 10.0])
        model = Net(p)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        tbmodel = tb.Trial(model, optim, loss, callbacks=[tb.callbacks.MostRecent(filepath='test.pt')],
                           pass_state=True).for_train_steps(training_steps)
        tbmodel.load_state_dict(torch.load('test.pt'))
        self.assertEqual(len(tbmodel.state[tb.HISTORY]), 2)
        self.assertAlmostEqual(model.pars[0].item(), 5.0, places=4)
        self.assertAlmostEqual(model.pars[1].item(), 0.0, places=4)
        self.assertAlmostEqual(model.pars[2].item(), 1.0, places=4)

        import os
        os.remove('test.pt')

    def test_only_model(self):
        p = torch.tensor([2.0, 1.0, 10.0])

        model = Net(p)

        tbmodel = tb.Trial(model)
        self.assertListEqual(tbmodel.run(), [])

    def test_no_model(self):
        tbmodel = tb.Trial(None)
        with self.assertWarns(Warning):
            tbmodel.run()
