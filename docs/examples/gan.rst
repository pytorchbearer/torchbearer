Training a GAN
====================================

We shall try to implement something more complicated using torchbearer - a Generative Adverserial Network (GAN).
This tutorial is a modified version of the GAN_ from the brilliant collection of GAN implementations PyTorch_GAN_ by eriklindernoren on github.

.. _PyTorch_GAN: https://github.com/eriklindernoren/PyTorch-GAN
.. _GAN: https://github.com/eriklindernoren/PyTorch-GAN#gan

Data and Constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first define all constants for the example.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 19-29

We then define a number of state keys for convenience. This is optional, however, it automatically avoids key conflicts.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 32-37

We then define the dataset and dataloader - for this example, MNIST.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 124-131

Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the generator and discriminator from PyTorch_GAN_ and combine them into a model that performs a single forward pass.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 86-102

Note that we have to be careful to remove the gradient information from the discriminator after doing the generator forward pass.

Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since our loss is complicated in this example, we shall forgo the basic loss criterion used in normal torchbearer models.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 159-160

Instead use a callback to provide the loss. Since this callback is very simple we can use callback decorators on a function (which takes state) to tell torchbearer when it should be called.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 105-112

Note that we have summed the separate discriminator and generator losses since their graphs are separated, this is allowable.

Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We would like to follow the discriminator and generator losses during training - note that we added these to state during the model definition.
We can then create metrics from these by decorating simple state fetcher metrics.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 139-146

Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can then train the torchbearer model on the GPU in the standard way.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 163-165

Visualising
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We borrow the image saving method from PyTorch_GAN_ and put it in a call back to save on training step - again using decorators.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 115-119

After 172400 iterations we see the following.

.. figure:: /_static/img/172400.png
   :scale: 200 %
   :alt: GAN generated samples after 172400 iterations


Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for the example is given below:

 :download:`Download Python source code: gan.py </_static/examples/gan.py>`



