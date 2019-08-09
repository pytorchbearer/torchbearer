<img src="https://raw.githubusercontent.com/pytorchbearer/torchbearer/master/docs/_static/img/logo_dark_text.svg?sanitize=true" width="100%"/>

[![PyPI version](https://badge.fury.io/py/torchbearer.svg)](https://badge.fury.io/py/torchbearer) [![Python 2.7 | 3.5 | 3.6 | 3.7](https://img.shields.io/badge/python-2.7%20%7C%203.5%20%7C%203.6%20%7C%203.7-brightgreen.svg)](https://www.python.org/) [![PyTorch 0.4.0 | 0.4.1 | 1.0.0 | 1.1.0](https://img.shields.io/badge/pytorch-0.4.0%20%7C%200.4.1%20%7C%201.0.0%20%7C%201.1.0-brightgreen.svg)](https://pytorch.org/) [![Build Status](https://travis-ci.com/pytorchbearer/torchbearer.svg?branch=master)](https://travis-ci.com/pytorchbearer/torchbearer) [![codecov](https://codecov.io/gh/pytorchbearer/torchbearer/branch/master/graph/badge.svg)](https://codecov.io/gh/pytorchbearer/torchbearer) [![Documentation Status](https://readthedocs.org/projects/torchbearer/badge/?version=latest)](https://torchbearer.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://pepy.tech/badge/torchbearer)](https://pepy.tech/project/torchbearer)

A model fitting library for PyTorch
## Contents
- [About](#about)
- [Notebooks](#notebooks)
- [Installation](#installation)
- [Citing Torchbearer](#citing)
- [Documentation](#docs)
- [Other Libraries](#others)

<a name="about"/>

## About

Torchbearer is a PyTorch model fitting library designed for use by researchers (or anyone really) working in deep learning or differentiable programming. Specifically, if you occasionally want to perform advanced custom operations but generally don't want to write hundreds of lines of untested code then this is the library for you.

<a name="notebooks"/>

## Notebooks

Below is a list of colab notebooks showing some of the things you can do with torchbearer.

### Quickstart

| | | |
| ---- | ---- | ---- |
| <img src="http://www.pytorchbearer.org/assets/img/examples/quickstart.jpg" width="128"> | **Quickstart:** Get up and running with torchbearer, training a simple CNN on CIFAR-10. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb) |
| <img src="http://www.pytorchbearer.org/assets/img/examples/callbacks.jpg" width="128"> | **Callbacks:** A detailed exploration of callbacks in torchbearer, with some useful visualisations. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb) |


| [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" width="32">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" width="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" width="32">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb) |
| ---- | ---- | ---- |

### Callbacks

| <img src="http://www.pytorchbearer.org/assets/img/examples/callbacks.jpg" width="128"> | A detailed exploration of callbacks in torchbearer, with some useful visualisations. |
| ---- | ---- |

| [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" width="32">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" width="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" width="32">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb) |
| ---- | ---- | ---- |

### Serialization

| This guide gives an introduction to serializing and restarting training in torchbearer. |
| ---- |

| [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" width="32">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/serialization.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" width="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/serialization.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" width="32">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/serialization.ipynb) |
| ---- | ---- | ---- |

### History and Replay

| This guide gives an introduction to the history returned by a trial and the ability to replay training. |
| ---- |

| [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" width="32">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/history.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" width="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/history.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" width="32">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/history.ipynb) |
| ---- | ---- | ---- |

### Custom Data Loaders

| This guide gives an introduction on how to run custom data loaders in torchbearer. |
| ---- |

| [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" width="32">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/custom_loaders.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" width="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/custom_loaders.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" width="32">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/custom_loaders.ipynb) |
| ---- | ---- | ---- |

### Data Parallel

| This guide gives an introduction to using torchbearer with DataParrallel. |
| ---- |

| [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" width="32">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/data_parallel.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" width="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/data_parallel.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" width="32">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/data_parallel.ipynb) |
| ---- | ---- | ---- |

### LiveLossPlot

| <img src="http://www.pytorchbearer.org/assets/img/examples/livelossplot.jpg" width="128"> | A demonstration of the LiveLossPlot callback included in torchbearer. |
| ---- | ---- |

| [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" width="32">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/livelossplot.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" width="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/livelossplot.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" width="32">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/livelossplot.ipynb) |
| ---- | ---- | ---- |

### PyCM

| <img src="http://www.pytorchbearer.org/assets/img/examples/pycm.jpg" width="128"> | A demonstration of the PyCM callback included in torchbearer for generating confusion matrices. |
| ---- | ---- |

| [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" width="32">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/pycm.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" width="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/pycm.ipynb) | [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" width="32">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/pycm.ipynb) |
| ---- | ---- | ---- |

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
- [Pytorch Lightning](https://github.com/williamFalcon/pytorch-lightning), lightweight wrapper on top of PyTorch with advanced multi-gpu, cluster and best practice support out-of-the-box.    
