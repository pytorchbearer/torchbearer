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
        state['est'] = self.pars
        return self.f()


def loss(y_pred, y_true):
    return y_pred


@tb.metrics.to_dict
class est(tb.metrics.Metric):
    def __init__(self):
        super().__init__('est')

    def process(self, state):
        return state['est'].data


steps = torch.tensor(list(range(50000)))
p = torch.tensor([2.0, 1.0, 10.0])

model = Net(p)
optim = torch.optim.SGD(model.parameters(), lr=0.0001)

tbmodel = tb.Model(model, optim, loss, [est(), 'loss'])
tbmodel.fit(steps, steps, 1, pass_state=True)
print(list(model.parameters())[0].data)
