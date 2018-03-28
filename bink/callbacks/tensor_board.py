import copy

from tensorboardX import SummaryWriter
from torch.autograd import Variable

from bink.callbacks import Callback

import os

import torch


class TensorBoard(Callback):
    def __init__(self, log_dir='./logs',
                 write_graph=True,
                 write_batch_metrics=False,
                 write_epoch_metrics=True):
        super(TensorBoard, self).__init__()

        self.log_dir = log_dir
        self.write_graph = write_graph

        if not write_batch_metrics:
            self.on_step_training = lambda _: ...
            self.on_step_validation = lambda _: ...

        if not write_epoch_metrics:
            self.on_end_epoch = lambda _: ...

        if self.write_graph:
            def handle_graph(state):
                dummy = Variable(torch.rand(state['x'].size()))
                model = copy.deepcopy(state['model'])
                if state['use_cuda']:
                    model = model.cpu()
                self._writer.add_graph(model, (dummy, ))
                self._handle_graph = lambda _: ...
            self._handle_graph = handle_graph
        else:
            self._handle_graph = lambda _: ...

        self._writer = None

    def on_start(self, state):
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(self.log_dir, state['model'].__class__.__name__ + '_' + current_time)
        self._writer = SummaryWriter(log_dir=log_dir)

    def on_end(self, state):
        self._writer.close() if self._writer is not None else ...

    def on_sample(self, state):
        self._handle_graph(state)

    def on_step_training(self, state):
        for metric in state['metrics']:
            self._writer.add_scalars('batch/' + metric, {'epoch ' + str(state['epoch']): state['metrics'][metric]}, state['t'])

    def on_step_validation(self, state):
        for metric in state['metrics']:
            self._writer.add_scalars('batch/' + metric, {'epoch ' + str(state['epoch']): state['metrics'][metric]}, state['t'])

    def on_end_epoch(self, state):
        for metric in state['metrics']:
            self._writer.add_scalar('epoch/' + metric, state['metrics'][metric], state['epoch'])