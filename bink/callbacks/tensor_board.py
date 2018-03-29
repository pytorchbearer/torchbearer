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
                 avg_data_channels=True,
                 write_data=True,
                 write_features=True,
                 features_key='y_pred'):
        self.log_dir = log_dir
        self.comment = comment
        self.num_images = num_images
        self.avg_pool_size = avg_pool_size
        self.avg_data_channels = avg_data_channels
        self.write_data = write_data
        self.write_features = write_features
        self.features_key = features_key

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

            data = None

            if state['epoch'] == 0 and self.write_data:
                if self.avg_data_channels:
                    data = torch.mean(x, 1)
                else:
                    data = x

                data = data.view(data.size(0), -1)

            feature = None

            if self.write_features:
                feature = state[self.features_key].data.clone()
                feature = feature.view(feature.size(0), -1)

            label = state['y_true'].data.clone()

            if state['t'] == 0:
                remaining = self.num_images if self.num_images < label.size(0) else label.size(0)

                self._images = x[:remaining].cpu()
                self._labels = label[:remaining].cpu()

                if data is not None:
                    self._data = data[:remaining].cpu()

                if feature is not None:
                    self._features = feature[:remaining].cpu()
            else:
                remaining = self.num_images - self._labels.size(0)

                if remaining > label.size(0):
                    remaining = label.size(0)

                self._images = torch.cat((self._images, x[:remaining].cpu()), dim=0)
                self._labels = torch.cat((self._labels, label[:remaining].cpu()), dim=0)

                if data is not None:
                    self._data = torch.cat((self._data, data[:remaining].cpu()), dim=0)

                if feature is not None:
                    self._features = torch.cat((self._features, feature[:remaining].cpu()), dim=0)

            if self._labels.size(0) >= self.num_images:
                if state['epoch'] == 0 and self.write_data:
                    self._writer.add_embedding(self._data, metadata=self._labels, label_img=self._images, tag='data', global_step=-1)
                if self.write_features:
                    self._writer.add_embedding(self._features, metadata=self._labels, label_img=self._images, tag='features', global_step=state['epoch'])
                self.done = True

    def on_end_epoch(self, state):
        if self.write_features:
            self.done = False

    def on_end(self, state):
        self._writer.close()
