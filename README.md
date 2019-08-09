<img src="https://raw.githubusercontent.com/pytorchbearer/torchbearer/master/docs/_static/img/logo_dark_text.svg?sanitize=true" width="100%"/>

[![PyPI version](https://badge.fury.io/py/torchbearer.svg)](https://badge.fury.io/py/torchbearer) [![Python 2.7 | 3.5 | 3.6 | 3.7](https://img.shields.io/badge/python-2.7%20%7C%203.5%20%7C%203.6%20%7C%203.7-brightgreen.svg)](https://www.python.org/) [![PyTorch 0.4.0 | 0.4.1 | 1.0.0 | 1.1.0](https://img.shields.io/badge/pytorch-0.4.0%20%7C%200.4.1%20%7C%201.0.0%20%7C%201.1.0-brightgreen.svg)](https://pytorch.org/) [![Build Status](https://travis-ci.com/pytorchbearer/torchbearer.svg?branch=master)](https://travis-ci.com/pytorchbearer/torchbearer) [![codecov](https://codecov.io/gh/pytorchbearer/torchbearer/branch/master/graph/badge.svg)](https://codecov.io/gh/pytorchbearer/torchbearer) [![Documentation Status](https://readthedocs.org/projects/torchbearer/badge/?version=latest)](https://torchbearer.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://pepy.tech/badge/torchbearer)](https://pepy.tech/project/torchbearer)

<p align="center">
  <a href="http://pytorchbearer.org">Website</a> •
  <a href="https://torchbearer.readthedocs.io/en/latest/">Docs</a> •
  <a href="#examples">Examples</a> •
  <a href="#install">Install</a> •
  <a href="#citing">Citing</a> •
  <a href="#related">Related</a>
</p>

<a id="about"></a>

A PyTorch model fitting library designed for use by researchers (or anyone really) working in deep learning or differentiable programming. Specifically, we aim to dramatically reduce the amount of boilerplate code you need to write without limiting the functionality and openness of PyTorch.

<a id="examples"></a>

## Examples

<a id="general"></a>

### General

| | | |
| ---- | ---- | ---- |
| <img src="http://www.pytorchbearer.org/assets/img/examples/quickstart.jpg" width="256"> | **Quickstart:** Get up and running with torchbearer, training a simple CNN on CIFAR-10. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb) |
| <img src="http://www.pytorchbearer.org/assets/img/examples/callbacks.jpg" width="256"> | **Callbacks:** A detailed exploration of callbacks in torchbearer, with some useful visualisations. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb) |
| | **Serialization:** This guide gives an introduction to serializing and restarting training in torchbearer. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/serialization.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/serialization.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/serialization.ipynb) |
| | **History and Replay:** This guide gives an introduction to the history returned by a trial and the ability to replay training. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/history.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/history.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/history.ipynb) |
| | **Custom Data Loaders:** This guide gives an introduction on how to run custom data loaders in torchbearer. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/custom_loaders.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/custom_loaders.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/custom_loaders.ipynb) |
| | **Data Parallel:** This guide gives an introduction to using torchbearer with DataParrallel. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/data_parallel.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/data_parallel.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/data_parallel.ipynb) |
| <img src="http://www.pytorchbearer.org/assets/img/examples/livelossplot.jpg" width="256"> | **LiveLossPlot:**  A demonstration of the LiveLossPlot callback included in torchbearer. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/livelossplot.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/livelossplot.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/livelossplot.ipynb) |
| <img src="http://www.pytorchbearer.org/assets/img/examples/pycm.jpg" width="256"> | **PyCM:**  A demonstration of the PyCM callback included in torchbearer for generating confusion matrices. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/pycm.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/pycm.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/pycm.ipynb) |

<a id="deep"></a>

### Deep Learning

| | | |
| ---- | ---- | ---- |
| <img src="http://www.pytorchbearer.org/assets/img/examples/vae.jpg" width="256"> | **Training a VAE:** A demonstration of how to train (add do a simple visualisation of) a Variational Auto-Encoder (VAE) on MNIST with torchbearer. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/vae.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/vae.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/vae.ipynb) |
| <img src="http://www.pytorchbearer.org/assets/img/examples/gan.jpg" width="256"> | **Training a GAN:** A demonstration of how to train (add do a simple visualisation of) a Generative Adversarial Network (GAN) on MNIST with torchbearer. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/gan.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/gan.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/gan.ipynb) |
| <img src="http://www.pytorchbearer.org/assets/img/examples/adversarial.jpg" width="256"> | **Generating Adversarial Examples:** A demonstration of how to perform a simple adversarial attack with torchbearer. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/adversarial.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/adversarial.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/adversarial.ipynb) |
| <img src="http://www.pytorchbearer.org/assets/img/examples/transfer.jpg" width="256"> | **Transfer Learning with Torchbearer:** A demonstration of how to perform transfer learning on STL10 with torchbearer. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/transfer_learning.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/transfer_learning.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/transfer_learning.ipynb) |
| <img src="http://www.pytorchbearer.org/assets/img/examples/regulariser.jpg" width="256"> | **Regularisers in Torchbearer:** A demonstration of how to use all of the built-in regularisers in torchbearer (Mixup, CutOut, CutMix, Random Erase, Label Smoothing and Sample Pairing). |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/regularisers.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/regularisers.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/regularisers.ipynb) |
| <img src="http://www.pytorchbearer.org/assets/img/examples/cam.jpg" width="256"> | **Class Appearance Model:** A demonstration of the Class Appearance Model (CAM) callback in torchbearer. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/cam.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/cam.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/cam.ipynb) |

<a id="diff"></a>

### Differentiable Programming

| | | |
| ---- | ---- | ---- |
| <img src="http://www.pytorchbearer.org/assets/img/examples/optimisers.jpg" width="256"> | **Optimising Functions:** An example (and some fun visualisations) showing how torchbearer can be used for the purpose of optimising functions with respect to their parameters using gradient descent. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/basic_opt.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/basic_opt.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/basic_opt.ipynb) |
| <img src="http://www.pytorchbearer.org/assets/img/examples/svm.jpg" width="256"> | **Linear SVM:** Train a linear support vector machine (SVM) using torchbearer, with an interactive visualisation! |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/svm_linear.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/svm_linear.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/svm_linear.ipynb) |
| <img src="http://www.pytorchbearer.org/assets/img/examples/amsgrad.jpg" width="256"> | **Breaking Adam:** The Adam optimiser doesn't always converge, in this example we reimplement some of the function optimisations from the AMSGrad paper showing this empirically. |  [<img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">](https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/amsgrad.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">](https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/amsgrad.ipynb) [<img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">](https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/amsgrad.ipynb) |

<a id="installation"></a>

## Install

The easiest way to install torchbearer is with pip:

`pip install torchbearer`

Alternatively, build from source with:

`pip install git+https://github.com/pytorchbearer/torchbearer`

<a id="citing"></a>

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

<a id="related"></a>

## Related

Torchbearer isn't the only library for training PyTorch models. Here are a few others that might better suit your needs (this is by no means a complete list, see the [awesome pytorch list](https://github.com/bharathgs/Awesome-pytorch-list) or [the incredible pytorch](https://github.com/ritchieng/the-incredible-pytorch) for more):
- [skorch](https://github.com/dnouri/skorch), model wrapper that enables use with scikit-learn - crossval etc. can be very useful
- [PyToune](https://github.com/GRAAL-Research/pytoune), simple Keras style API
- [ignite](https://github.com/pytorch/ignite), advanced model training from the makers of PyTorch, can need a lot of code for advanced functions (e.g. Tensorboard)
- [TorchNetTwo (TNT)](https://github.com/pytorch/tnt), can be complex to use but well established, somewhat replaced by ignite
- [Inferno](https://github.com/inferno-pytorch/inferno), training utilities and convenience classes for PyTorch   
- [Pytorch Lightning](https://github.com/williamFalcon/pytorch-lightning), lightweight wrapper on top of PyTorch with advanced multi-gpu, cluster and best practice support out-of-the-box.    
