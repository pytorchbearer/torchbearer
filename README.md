**Note:**
We're moving to PyTorch Lightning! Read about the move [here](https://medium.com/pytorch/pytorch-frameworks-unite-torchbearer-joins-pytorch-lightning-c588e1e68c98). From the end of February, torchbearer will no longer be actively maintained. We'll continue to fix bugs when they are found and ensure that torchbearer runs on new versions of pytorch. However, we won't plan or implement any new functionality (if there's something you'd like to see in a training library, consider creating an issue on [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)).

<img alt="logo" src="https://raw.githubusercontent.com/pytorchbearer/torchbearer/master/docs/_static/img/logo_dark_text.svg?sanitize=true" width="100%"/>

[![PyPI version](https://badge.fury.io/py/torchbearer.svg)](https://badge.fury.io/py/torchbearer) [![Python 2.7 | 3.5 | 3.6 | 3.7](https://img.shields.io/badge/python-2.7%20%7C%203.5%20%7C%203.6%20%7C%203.7-brightgreen.svg)](https://www.python.org/) [![PyTorch 1.0.0 | 1.1.0 | 1.2.0 | 1.3.0 | 1.4.0](https://img.shields.io/badge/pytorch-1.0.0%20%7C%201.1.0%20%7C%201.2.0%20%7C%201.3.0%20%7C%201.4.0-brightgreen.svg)](https://pytorch.org/) [![Build Status](https://travis-ci.com/pytorchbearer/torchbearer.svg?branch=master)](https://travis-ci.com/pytorchbearer/torchbearer) [![codecov](https://codecov.io/gh/pytorchbearer/torchbearer/branch/master/graph/badge.svg)](https://codecov.io/gh/pytorchbearer/torchbearer) [![Documentation Status](https://readthedocs.org/projects/torchbearer/badge/?version=latest)](https://torchbearer.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://pepy.tech/badge/torchbearer)](https://pepy.tech/project/torchbearer)

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

<table>
    <tr>
        <td rowspan="3" width="160">
            <img src="http://www.pytorchbearer.org/assets/img/examples/quickstart.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>Quickstart:</b> Get up and running with torchbearer, training a simple CNN on CIFAR-10.
        </td>
        <td align="center" width="80">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3">
            <img src="http://www.pytorchbearer.org/assets/img/examples/callbacks.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>Callbacks:</b> A detailed exploration of callbacks in torchbearer, with some useful visualisations.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3">
            <img src="http://www.pytorchbearer.org/assets/img/examples/imaging.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>Imaging:</b> A detailed exploration of the imaging sub-package in torchbearer, useful for showing visualisations during training.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/imaging.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/imaging.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/imaging.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3" colspan="2">
            <b>Serialization:</b> This guide gives an introduction to serializing and restarting training in torchbearer.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/serialization.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/serialization.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/serialization.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3" colspan="2">
            <b>History and Replay:</b> This guide gives an introduction to the history returned by a trial and the ability to replay training.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/history.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/history.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/history.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3" colspan="2">
            <b>Custom Data Loaders:</b> This guide gives an introduction on how to run custom data loaders in torchbearer.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/custom_loaders.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/custom_loaders.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/custom_loaders.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3" colspan="2">
            <b>Data Parallel:</b> This guide gives an introduction to using torchbearer with DataParrallel.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/data_parallel.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/data_parallel.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/data_parallel.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3">
            <img src="http://www.pytorchbearer.org/assets/img/examples/livelossplot.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>LiveLossPlot:</b> A demonstration of the LiveLossPlot callback included in torchbearer.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/livelossplot.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/livelossplot.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/livelossplot.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3">
            <img src="http://www.pytorchbearer.org/assets/img/examples/pycm.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>PyCM:</b> A demonstration of the PyCM callback included in torchbearer for generating confusion matrices.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/pycm.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/pycm.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/pycm.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3" colspan="2">
            <b>NVIDIA Apex:</b> A guide showing how to perform half and mixed precision training in torchbearer with NVIDIA Apex.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/apex_torchbearer.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/apex_torchbearer.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/apex_torchbearer.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
</table>

<a id="deep"></a>

### Deep Learning

<table>
    <tr>
        <td rowspan="3" width="160">
            <img src="http://www.pytorchbearer.org/assets/img/examples/vae.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>Training a VAE:</b> A demonstration of how to train (add do a simple visualisation of) a Variational Auto-Encoder (VAE) on MNIST with torchbearer.
        </td>
        <td align="center" width="80">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/vae.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/vae.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/vae.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3">
            <img src="http://www.pytorchbearer.org/assets/img/examples/gan.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>Training a GAN:</b> A demonstration of how to train (add do a simple visualisation of) a Generative Adversarial Network (GAN) on MNIST with torchbearer.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/gan.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/gan.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/gan.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3">
            <img src="http://www.pytorchbearer.org/assets/img/examples/adversarial.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>Generating Adversarial Examples:</b> A demonstration of how to perform a simple adversarial attack with torchbearer.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/adversarial.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/adversarial.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/adversarial.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3">
            <img src="http://www.pytorchbearer.org/assets/img/examples/transfer.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>Transfer Learning with Torchbearer:</b> A demonstration of how to perform transfer learning on STL10 with torchbearer.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/transfer_learning.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/transfer_learning.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/transfer_learning.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3">
            <img src="http://www.pytorchbearer.org/assets/img/examples/regulariser.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>Regularisers in Torchbearer:</b> A demonstration of how to use all of the built-in regularisers in torchbearer (Mixup, CutOut, CutMix, Random Erase, Label Smoothing and Sample Pairing).
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/regularisers.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
        <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/regularisers.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/regularisers.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3" colspan="2">
            <b>Manifold Mixup:</b> A demonstration of how to use the Manifold Mixup callback in Torchbearer.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/manifold_mixup.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/manifold_mixup.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/manifold_mixup.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3">
            <img src="http://www.pytorchbearer.org/assets/img/examples/cam.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>Class Appearance Model:</b> A demonstration of the Class Appearance Model (CAM) callback in torchbearer.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/cam.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/cam.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/cam.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
</table>

<a id="diff"></a>

### Differentiable Programming

<table>
    <tr>
        <td rowspan="3" width="160">
            <img src="http://www.pytorchbearer.org/assets/img/examples/optimisers.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>Optimising Functions:</b> An example (and some fun visualisations) showing how torchbearer can be used for the purpose of optimising functions with respect to their parameters using gradient descent.
        </td>
        <td align="center" width="80">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/basic_opt.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/basic_opt.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/basic_opt.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3">
            <img src="http://www.pytorchbearer.org/assets/img/examples/svm.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>Linear SVM:</b> Train a linear support vector machine (SVM) using torchbearer, with an interactive visualisation!
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/svm_linear.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/svm_linear.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/svm_linear.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
    <tr>
        <td rowspan="3">
            <img src="http://www.pytorchbearer.org/assets/img/examples/amsgrad.jpg" width="256">
        </td>    
        <td rowspan="3">
            <b>Breaking Adam:</b> The Adam optimiser doesn't always converge, in this example we reimplement some of the function optimisations from the AMSGrad paper showing this empirically.
        </td>
        <td align="center">
            <a href="https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/amsgrad.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/nbviewer_logo.svg" height="34">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://github.com/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/amsgrad.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/github_logo.png" height="32">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">
            <a href="https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/amsgrad.ipynb">
                <img src="http://www.pytorchbearer.org/assets/img/colab_logo.png" height="28">
            </a>
        </td>
    </tr>
</table>

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
- [Pytorch Lightning](https://github.com/williamFalcon/pytorch-lightning), lightweight wrapper on top of PyTorch with advanced multi-gpu and cluster support
- [Pywick](https://github.com/achaiah/pywick), high-level training framework, based on torchsample, support for various segmentation models
