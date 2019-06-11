<img src="https://raw.githubusercontent.com/pytorchbearer/torchbearer/master/docs/_static/img/logo_dark_text.svg?sanitize=true" width="100%"/>

[![PyPI version](https://badge.fury.io/py/torchbearer.svg)](https://badge.fury.io/py/torchbearer) [![Python 2.7 | 3.5 | 3.6 | 3.7](https://img.shields.io/badge/python-2.7%20%7C%203.5%20%7C%203.6%20%7C%203.7-brightgreen.svg)](https://www.python.org/) [![PyTorch 0.4.0 | 0.4.1 | 1.0.0 | 1.1.0](https://img.shields.io/badge/pytorch-0.4.0%20%7C%200.4.1%20%7C%201.0.0%20%7C%201.1.0-brightgreen.svg)](https://pytorch.org/) [![Build Status](https://travis-ci.com/pytorchbearer/torchbearer.svg?branch=master)](https://travis-ci.com/pytorchbearer/torchbearer) [![codecov](https://codecov.io/gh/pytorchbearer/torchbearer/branch/master/graph/badge.svg)](https://codecov.io/gh/pytorchbearer/torchbearer) [![Documentation Status](https://readthedocs.org/projects/torchbearer/badge/?version=latest)](https://torchbearer.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://pepy.tech/badge/torchbearer)](https://pepy.tech/project/torchbearer)

A model fitting library for PyTorch
## Contents
- [About](#about)
- [Key Features](#features)
- [Installation](#installation)
- [Citing Torchbearer](#citing)
- [Quickstart](#quick)
- [Documentation](#docs)
- [Other Libraries](#others)

![SVM fitting](https://raw.githubusercontent.com/pytorchbearer/torchbearer/master/docs/_static/img/svm_fit.gif)![GAN Gif](https://raw.githubusercontent.com/pytorchbearer/torchbearer/master/docs/_static/img/gan.gif)

<a name="about"/>

## About

Torchbearer is a PyTorch model fitting library designed for use by researchers (or anyone really) working in deep learning or differentiable programming. Specifically, if you occasionally want to perform advanced custom operations but generally don't want to write hundreds of lines of untested code then this is the library for you.

Above are a linear SVM (differentiable program) visualisation from the [docs](http://torchbearer.readthedocs.io/en/latest/examples/svm_linear.html) in less than 100 lines of code and a GAN visualisation from the [docs](http://torchbearer.readthedocs.io/en/latest/examples/gan.html) both implemented using torchbearer and pytorch.

<a name="features"/>

## Key Features

- Model fitting API using calls to [run(...)](http://torchbearer.readthedocs.io/en/latest/code/main.html#torchbearer.torchbearer.Trial.run) on Trial instances which are saveable, resumable and replayable
- Sophisticated [metric API](http://torchbearer.readthedocs.io/en/latest/code/metrics.html) which supports calculation data (e.g. accuracy) flowing to multiple aggregators which can calculate running values (e.g. mean) and values for the epoch (e.g. std, mean, area under curve)
- Default accuracy metric which infers the accuracy to use from the criterion
- Simple [callback API](http://torchbearer.readthedocs.io/en/latest/code/callbacks.html) with a persistent model state that supports adding to the loss or accessing the metric values
- A host of callbacks included from the start that enable: [tensorboard and visdom logging](http://torchbearer.readthedocs.io/en/latest/code/callbacks.html#module-torchbearer.callbacks.tensor_board) (for metrics, images and data), [model checkpointing](http://torchbearer.readthedocs.io/en/latest/code/callbacks.html#module-torchbearer.callbacks.checkpointers), [weight decay](http://torchbearer.readthedocs.io/en/latest/code/callbacks.html#module-torchbearer.callbacks.weight_decay), [learning rate schedulers](http://torchbearer.readthedocs.io/en/latest/code/callbacks.html#module-torchbearer.callbacks.torch_scheduler), [gradient clipping](http://torchbearer.readthedocs.io/en/latest/code/callbacks.html#module-torchbearer.callbacks.gradient_clipping) and more
- Decorator APIs for [metrics](http://torchbearer.readthedocs.io/en/latest/code/metrics.html#module-torchbearer.metrics.decorators) and [callbacks](http://torchbearer.readthedocs.io/en/latest/code/callbacks.html#module-torchbearer.callbacks.decorators) that allow for simple construction
- An [example library](http://torchbearer.readthedocs.io/en/latest/examples/quickstart.html) with a set of demos showing how complex deep learning models (such as [GANs](http://torchbearer.readthedocs.io/en/latest/examples/gan.html) and [VAEs](http://torchbearer.readthedocs.io/en/latest/examples/vae.html)) and differentiable programs (like [SVMs](http://torchbearer.readthedocs.io/en/latest/examples/svm_linear.html)) can be implemented easily with torchbearer
- Fully tested; as researchers we want to trust that our metrics and callbacks work properly, we have therefore tested everything thoroughly for peace of mind

<a name="installation"/>

## Installation

The easiest way to install torchbearer is with pip:

`pip install torchbearer`

Alternatively, build from source with:

`pip install git+https://github.com/pytorchbearer/torchbearer`

<a name="citing"/>

## Citing Torchbearer

If you find that torchbearer is useful to your research then please consider citing our preprint: [Torchbearer: A Model Fitting Library for PyTorch](https://arxiv.org/abs/1809.03363), with the following BibTeX entry:

```
@article{torchbearer2018,
  author = {Ethan Harris and Matthew Painter and Jonathon Hare},
  title = {Torchbearer: A Model Fitting Library for PyTorch},
  journal  = {arXiv preprint arXiv:1809.03363},
  year = {2018}
}
```

<a name="quick"/>

## Quickstart

- Define your data and model as usual (here we use a simple CNN on Cifar10). Note that we use torchbearers DatasetValidationSplitter here to create a validation set (10% of the data). This is essential to avoid [over-fitting to your test data](http://blog.kaggle.com/2012/07/06/the-dangers-of-overfitting-psychopathy-post-mortem/):

```python
BATCH_SIZE = 128

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

dataset = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), normalize]))
splitter = DatasetValidationSplitter(len(dataset), 0.1)
trainset = splitter.get_train_dataset(dataset)
valset = splitter.get_val_dataset(dataset)

traingen = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
valgen = torch.utils.data.DataLoader(valset, pin_memory=True, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)


testset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False, download=True,
                                       transform=transforms.Compose([transforms.ToTensor(), normalize]))
testgen = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 16, stride=2, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, stride=2, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, stride=2, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.classifier = nn.Linear(576, 10)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 576)
        return self.classifier(x)


model = SimpleModel()
```

- Now that we have a model we can train it simply by wrapping it in a torchbearer Trial instance:

```python
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
loss = nn.CrossEntropyLoss()

from torchbearer import Trial

trial = Trial(model, optimizer, criterion=loss, metrics=['acc', 'loss']).to('cuda')
trial = trial.with_generators(train_generator=traingen, val_generator=valgen, test_generator=testgen)
trial.run(epochs=10)

trial.evaluate(data_key=torchbearer.TEST_DATA)
```
- Running that code gives output using Tqdm and providing running accuracies and losses during the training phase:

```
0/10(t): 100%|██████████| 352/352 [00:02<00:00, 163.98it/s, acc=0.4339, loss=1.5776, running_acc=0.5202, running_loss=1.3494]
0/10(v): 100%|██████████| 40/40 [00:00<00:00, 365.42it/s, val_acc=0.5266, val_loss=1.3208]
.
.
.
9/10(t): 100%|██████████| 352/352 [00:02<00:00, 165.28it/s, acc=0.7515, loss=0.715, running_acc=0.7352, running_loss=0.7492]
9/10(v): 100%|██████████| 40/40 [00:00<00:00, 310.76it/s, val_acc=0.6792, val_loss=0.9743]
0/1(e): 100%|██████████| 79/79 [00:00<00:00, 233.06it/s, test_acc=0.6673, test_loss=0.9741]
```

<a name="docs"/>

## Documentation

Our documentation containing the API reference, examples and notes can be found at [torchbearer.readthedocs.io](https://torchbearer.readthedocs.io)

<a name="others"/>

## Other Libraries

Torchbearer isn't the only library for training PyTorch models. Here are a few others that might better suit your needs (this is by no means a complete list, see the [awesome pytorch list](https://github.com/bharathgs/Awesome-pytorch-list) or [the incredible pytorch](https://github.com/ritchieng/the-incredible-pytorch) for more):
- [skorch](https://github.com/dnouri/skorch), model wrapper that enables use with scikit-learn - crossval etc. can be very useful
- [PyToune](https://github.com/GRAAL-Research/pytoune), simple Keras style API
- [ignite](https://github.com/pytorch/ignite), advanced model training from the makers of PyTorch, can need a lot of code for advanced functions (e.g. Tensorboard)
- [TorchNetTwo (TNT)](https://github.com/pytorch/tnt), can be complex to use but well established, somewhat replaced by ignite
- [Inferno](https://github.com/inferno-pytorch/inferno), training utilities and convenience classes for PyTorch
