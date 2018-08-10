import torch
from torch.nn import Module

import torchbearer as tb
from torchbearer.callbacks import TensorBoard

import random


class Online(Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.zeros(1))

    def forward(self, _, state):
        """
        function to be minimised:
        f(x) = 1010x if t mod 101 = 1, else -10x
        """
        if state[tb.BATCH] % 101 == 1:
            res = 1010 * self.x
        else:
            res = -10 * self.x

        return res


class Stochastic(Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.zeros(1))

    def forward(self, _):
        """
        function to be minimised:
        f(x) = 1010x with probability 0.01, else -10x
        """
        if random.random() <= 0.01:
            res = 1010 * self.x
        else:
            res = -10 * self.x

        return res


def loss(y_pred, _):
    return y_pred


@tb.metrics.to_dict
class est(tb.metrics.Metric):
    def __init__(self):
        super().__init__('est')

    def process(self, state):
        return state[tb.MODEL].x.data


@tb.callbacks.on_step_training
def greedy_update(state):
    if state[tb.MODEL].x > 1:
        state[tb.MODEL].x.data.fill_(1)
    elif state[tb.MODEL].x < -1:
        state[tb.MODEL].x.data.fill_(-1)


training_steps = 6000000

model = Online()

optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.99])
tbmodel = tb.Model(model, optim, loss, [est()])
tbmodel.fit_generator(None, pass_state=True, train_steps=training_steps, callbacks=[greedy_update, TensorBoard(comment='adam', write_graph=False, write_batch_metrics=True, write_epoch_metrics=False)])

optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.99], amsgrad=True)
tbmodel = tb.Model(model, optim, loss, [est()])
tbmodel.fit_generator(None, pass_state=True, train_steps=training_steps, callbacks=[greedy_update, TensorBoard(comment='amsgrad', write_graph=False, write_batch_metrics=True, write_epoch_metrics=False)])

model = Stochastic()

optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.99])
tbmodel = tb.Model(model, optim, loss, [est()])
tbmodel.fit_generator(None, train_steps=training_steps, callbacks=[greedy_update, TensorBoard(comment='adam', write_graph=False, write_batch_metrics=True, write_epoch_metrics=False)])

optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.99], amsgrad=True)
tbmodel = tb.Model(model, optim, loss, [est()])
tbmodel.fit_generator(None, train_steps=training_steps, callbacks=[greedy_update, TensorBoard(comment='amsgrad', write_graph=False, write_batch_metrics=True, write_epoch_metrics=False)])
