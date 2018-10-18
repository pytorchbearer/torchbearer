import torch
from torch.nn import Module

import numpy as np

import torchbearer as tb

ESTIMATE = tb.state_key('est')


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
        state[ESTIMATE] = np.round(self.pars.detach().cpu().numpy(), 4)
        return self.f()


def loss(y_pred, y_true):
    return y_pred


p = torch.tensor([2.0, 1.0, 10.0])
training_steps = 50000

model = Net(p)
optim = torch.optim.SGD(model.parameters(), lr=0.0001)

tbtrial = tb.Trial(model, optim, loss, [ESTIMATE, 'loss'], pass_state=True).for_train_steps(training_steps).to('cuda')
tbtrial.run()
print(list(model.parameters())[0].data)
