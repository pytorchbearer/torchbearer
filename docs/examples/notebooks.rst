Notebooks List
================================
Here we have a list of example notebooks using Torchbearer with a brief description of the contents and broken down by broad subject.


.. |colab| image:: /_static/img/colab.jpg
    :width: 25

.. |nbviewer| image:: /_static/img/nbviewer_logo.svg
    :width: 12

General
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Quickstart Guide**:

    This guide will give a quick intro to training PyTorch models with Torchbearer.

    |nbviewer| `Preview <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb>`__   :download:`Download Notebook </_static/notebooks/quickstart.ipynb>`   |colab| `Run on Colab <https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/quickstart.ipynb>`__

- **Callbacks Guide**:

    This guide will give an introduction to using callbacks with Torchbearer.

    |nbviewer| `Preview <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb>`__   :download:`Download Notebook </_static/notebooks/callbacks.ipynb>`   |colab| `Run on Colab <https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb>`__

Deep Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Training a VAE**:

    This guide covers training a variational auto-encoder (VAE) in Torchbearer, taking advantage of the persistent state.

    |nbviewer| `Preview <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/vae.ipynb>`__   :download:`Download Notebook </_static/notebooks/vae.ipynb>`   |colab| `Run on Colab <https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/vae.ipynb>`__

- **Training a GAN**:

    This guide will cover how to train a Generative Adversarial Network (GAN) in Torchbearer using custom closures to allow for the more complicated training loop.

    |nbviewer| `Preview <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/gan.ipynb>`__   :download:`Download Notebook </_static/notebooks/gan.ipynb>`   |colab| `Run on Colab <https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/gan.ipynb>`__

- **Class Appearance Model**:

    In this example we will demonstrate the `ClassAppearanceModel <https://torchbearer.readthedocs.io/en/latest/code/callbacks.html#torchbearer.callbacks.imaging.inside_cnns.ClassAppearanceModel>`__ callback included in torchbearer. This implements
    one of the most simple (and therefore not always the most successful) deep visualisation techniques, discussed in the
    paper `Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps <https://arxiv.org/abs/1312.6034>`__

    |nbviewer| `Preview <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/cam.ipynb>`__   :download:`Download Notebook </_static/notebooks/cam.ipynb>`   |colab| `Run on Colab <https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/cam.ipynb>`__

- **Adversarial Example Generation**:

    This guide will cover how to perform a simple adversarial attack in Torchbearer.

    |nbviewer| `Preview <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/adversarial.ipynb>`__   :download:`Download Notebook </_static/notebooks/adversarial.ipynb>`   |colab| `Run on Colab <https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/adversarial.ipynb>`__


- **Transfer Learning**:

    This guide will cover how to perform transfer learning of a model with Torchbearer.

    |nbviewer| `Preview <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/transfer_learning.ipynb>`__   :download:`Download Notebook </_static/notebooks/transfer_learning.ipynb>`   |colab| `Run on Colab <https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/transfer_learning.ipynb>`__

Differentiable Programming
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Optimising Functions**:

    This guide will briefly show how we can do function optimisation using Torchbearer.

    |nbviewer| `Preview <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/basic_opt.ipynb>`__   :download:`Download Notebook </_static/notebooks/basic_opt.ipynb>`   |colab| `Run on Colab <https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/basic_opt.ipynb>`__

- **Linear SVM**:

    This guide will train a linear support vector machine (SVM) using Torchbearer.

    |nbviewer| `Preview <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/svm_linear.ipynb>`__   :download:`Download Notebook </_static/notebooks/svm_linear.ipynb>`   |colab| `Run on Colab <https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/svm_linear.ipynb>`__

- **Breaking ADAM**:

    This guide uses Torchbearer to implement `On the Convergence of Adam and Beyond <https://openreview.net/forum?id=ryQu7f-RZ>`__, one of the top papers at ICLR 2018, which demonstrated a case where ADAM does not converge.

    |nbviewer| `Preview <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/amsgrad.ipynb>`__   :download:`Download Notebook </_static/notebooks/amsgrad.ipynb>`   |colab| `Run on Colab <https://colab.research.google.com/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/amsgrad.ipynb>`__
