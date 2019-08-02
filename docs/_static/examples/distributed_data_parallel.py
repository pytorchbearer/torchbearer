import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
import torchbearer
import platform
from torchvision import datasets, transforms
import argparse


parser = argparse.ArgumentParser(description='Torchbearer Distributed Data Parallel MNIST')
parser.add_argument('--master-addr', '--master', '--host', '-m', dest='master', help='Address of master node')
parser.add_argument('--rank', '-r', dest='rank', help='Rank of this process')
parser.add_argument('--world-size', dest='world_size', default=2, help='World size')
args = parser.parse_args()


def setup():
    os.environ['MASTER_ADDR'] = args.master
    os.environ['MASTER_PORT'] = '29500'

    # initialize the process group
    dist.init_process_group("gloo", rank=args.rank, world_size=args.world_size)

    # Explicitly setting seed makes sure that models created in two processes
    # start from same random weights and biases. Alternatively, sync models
    # on start with the callback below.
    #torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(784, 100)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(100, 10)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def sync_model(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


@torchbearer.callbacks.on_init
def sync(state):
    sync_model(state[torchbearer.MODEL])


@torchbearer.callbacks.on_backward
def grad(state):
    average_gradients(state[torchbearer.MODEL])


@torchbearer.callbacks.on_sample
def flatten(state):
    state[torchbearer.X] = state[torchbearer.X].view(state[torchbearer.X].shape[0], -1)


def worker():
    setup()
    print("Rank and node: {}-{}".format(args.rank, platform.node()))

    model = ToyModel().to('cpu')
    ddp_model = DDP(model)

    kwargs = {}

    ds = datasets.MNIST('./data/mnist/', train=True, download=True,
         transform=transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))
          ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(ds)
    train_loader = torch.utils.data.DataLoader(ds,
        batch_size=128, sampler=train_sampler, **kwargs)

    test_ds = datasets.MNIST('./data/mnist', train=False,
              transform=transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]))
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
    test_loader = torch.utils.data.DataLoader(test_ds,
        batch_size=128, sampler=test_sampler,  **kwargs)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    trial = torchbearer.Trial(ddp_model, optimizer, loss_fn, metrics=['loss', 'acc'],
        callbacks=[sync, grad, flatten])
    trial.with_train_generator(train_loader)
    trial.run(10, verbose=2)

    print("Model hash: {}".format(hash(model)))
    print('First parameter: {}'.format(next(model.parameters())))

    cleanup()


if __name__ == "__main__":
    worker()
    print('done')
