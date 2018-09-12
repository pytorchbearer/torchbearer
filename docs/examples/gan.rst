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

We then define a number of state keys for convenience using :func:`.state_key`. This is optional, however, it automatically avoids key conflicts.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 32-37

We then define the dataset and dataloader - for this example, MNIST.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 124-132

Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the generator and discriminator from PyTorch_GAN_ and combine them into a model that performs a single forward pass.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 86-102

Note that we have to be careful to remove the gradient information from the discriminator after doing the generator forward pass.

Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since our loss computation in this example is complicated, we shall forgo the basic loss criterion used in normal torchbearer trials.
Instead we use a callback to provide the loss, in this case we use the :func:`.add_to_loss` callback decorator.
This decorates a function that returns a loss and automatically adds this to the main loss in training and validation.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 105-111

Note that we have summed the separate discriminator and generator losses, since their graphs are separated, this is allowable.

Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We would like to follow the discriminator and generator losses during training - note that we added these to state during the model definition.
We can then create metrics from these by decorating simple state fetcher metrics.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 140-147

Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can then train the torchbearer trial on the GPU in the standard way.
Note that when torchbearer is passed a ``None`` criterion it automatically sets the base loss to 0.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 160-164

Visualising
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We borrow the image saving method from PyTorch_GAN_ and put it in a call back to save :func:`~torchbearer.callbacks.decorators.on_step_training`.
We generate from the same inputs each time to get a better visulisation.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 114-120

Here is a Gif created from the saved images.

.. figure:: /_static/img/gan.gif
   :scale: 100 %
   :alt: GAN generated samples


Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for the example is given below:

 :download:`Download Python source code: gan.py </_static/examples/gan.py>`



