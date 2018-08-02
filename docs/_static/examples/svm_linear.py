# Based on svm-pytorch (https://github.com/kazuto1011/svm-pytorch)

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.animation import ImageMagickWriter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets.samples_generator import make_blobs

import torchbearer
import torchbearer.callbacks as callbacks
from torchbearer import Model
from torchbearer.callbacks import L2WeightDecay, ExponentialLR


class LinearSVM(nn.Module):
    """Support Vector Machine"""

    def __init__(self):
        super(LinearSVM, self).__init__()
        self.w = nn.Parameter(torch.randn(1, 2), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        h = x.matmul(self.w.t()) + self.b
        return h


def hinge_loss(y_pred, y_true):
    return torch.mean(torch.clamp(1 - y_pred.t() * y_true, min=0))


X, Y = make_blobs(n_samples=1024, centers=2, cluster_std=1.2, random_state=1)
X = (X - X.mean()) / X.std()
Y[np.where(Y == 0)] = -1
X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)


delta = 0.01
x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
x, y = np.meshgrid(x, y)
xy = list(map(np.ravel, [x, y]))


def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw_idle()
            canvas.start_event_loop(interval)
            return


@callbacks.on_start
def scatter(_):
    plt.figure(figsize=(5, 5))
    plt.ion()
    plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)


@callbacks.on_step_training
def draw_margin(state):
    if state[torchbearer.BATCH] % 10 == 0:
        w = svm.w[0].detach().to('cpu').numpy()
        b = svm.b[0].detach().to('cpu').numpy()

        z = (w.dot(xy) + b).reshape(x.shape)
        z[np.where(z > 1.)] = 4
        z[np.where((z > 0.) & (z <= 1.))] = 3
        z[np.where((z > -1.) & (z <= 0.))] = 2
        z[np.where(z <= -1.)] = 1

        # plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
        # plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
        if 'contour' in state:
            for coll in state['contour'].collections:
                coll.remove()
            state['contour'] = plt.contourf(x, y, z, cmap=plt.cm.jet, alpha=0.5)
            # state['contour'].cmap.set_under((1., 1., 1., 0.))
            # state['contour'].set_clim(0, 4)
        else:
            state['contour'] = plt.contourf(x, y, z, cmap=plt.cm.jet, alpha=0.5)
            # state['contour'].cmap.set_under((1., 1., 1., 0.))
            # state['contour'].set_clim(0, 4)
            plt.tight_layout()
            plt.show()

            state['writer'] = ImageMagickWriter(fps=7)
            state['writer'].setup(plt.gcf(), 'svm_fit.gif', dpi=80)
        # plt.draw()
        mypause(0.001)
        # if state[torchbearer.BATCH] % 30 == 0:
        state['writer'].grab_frame()


svm = LinearSVM()
model = Model(svm, optim.SGD(svm.parameters(), 0.1), hinge_loss, ['loss']).to('cuda')

state = model.fit(X, Y, batch_size=32, epochs=20, callbacks=[scatter, draw_margin, ExponentialLR(0.999, step_on_batch=True), L2WeightDecay(0.01, params=[svm.W])])

# plt.ioff()
# plt.show()
state['writer'].finish()
