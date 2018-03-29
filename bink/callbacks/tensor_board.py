import copy

from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F

from bink.callbacks import Callback

import os

import torch


class TensorBoard(Callback):
    def __init__(self, log_dir='./logs',
                 write_graph=True,
                 write_batch_metrics=False,
                 write_epoch_metrics=True,
                 comment='bink'):
        super(TensorBoard, self).__init__()

        self.log_dir = log_dir
        self.write_graph = write_graph
        self.write_batch_metrics = write_batch_metrics
        self.comment = comment

        if not write_epoch_metrics:
            self.on_end_epoch = lambda _: ...

        if self.write_graph:
            def handle_graph(state):
                dummy = Variable(torch.rand(state['x'].size()), volatile=True)
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
        log_dir = os.path.join(self.log_dir, state['model'].__class__.__name__ + '_' + self.comment)
        self._writer = SummaryWriter(log_dir=log_dir)

    def on_end(self, state):
        self._writer.close()

    def on_sample(self, state):
        self._handle_graph(state)

    def on_step_training(self, state):
        if self.write_batch_metrics:
            for metric in state['metrics']:
                self._writer.add_scalars('batch/' + metric, {'epoch ' + str(state['epoch']): state['metrics'][metric]}, state['t'])

    def on_step_validation(self, state):
        if self.write_batch_metrics:
            for metric in state['metrics']:
                self._writer.add_scalars('batch/' + metric, {'epoch ' + str(state['epoch']): state['metrics'][metric]}, state['t'])

    def on_end_epoch(self, state):
        for metric in state['metrics']:
            self._writer.add_scalar('epoch/' + metric, state['metrics'][metric], state['epoch'])


class TensorBoardImageVis(Callback):
    def __init__(self, log_dir='./logs',
                 comment='bink',
                 num_images=100,
                 avg_pool_size=1,
                 avg_feature_channels=True):
        self.log_dir = log_dir
        self.comment = comment
        self.num_images = num_images
        self.avg_pool_size = avg_pool_size
        self.avg_feature_channels = avg_feature_channels

        self._writer = None

        self.done = False

    def on_start(self, state):
        log_dir = os.path.join(self.log_dir, state['model'].__class__.__name__ + '_' + self.comment)
        self._writer = SummaryWriter(log_dir=log_dir)

    def on_step_validation(self, state):
        if not self.done:

            x = state['x'].data.clone()

            if len(x.size()) == 3:
                x = x.unsqueeze(1)

            x = F.avg_pool2d(Variable(x, volatile=True), self.avg_pool_size).data

            if self.avg_feature_channels:
                feature = torch.mean(x, 1)
            else:
                feature = x

            feature = feature.view(feature.size(0), -1)
            label = state['y_true'].data.clone()

            if state['t'] == 0:

                remaining = self.num_images if self.num_images < feature.size(0) else feature.size(0)

                self._features = feature[:remaining].cpu()
                self._images = x[:remaining].cpu()
                self._labels = label[:remaining].cpu()
            else:

                remaining = self.num_images - self._features.size(0)

                if remaining > feature.size(0):
                    remaining = feature.size(0)

                self._features = torch.cat((self._features, feature[:remaining].cpu()), dim=0)
                self._images = torch.cat((self._images, x[:remaining].cpu()), dim=0)
                self._labels = torch.cat((self._labels, label[:remaining].cpu()), dim=0)

            if self._features.size(0) >= self.num_images:
                self._writer.add_embedding(self._features, metadata=self._labels, label_img=self._images)
                self.done = True

    def on_end(self, state):
        self._writer.close()
