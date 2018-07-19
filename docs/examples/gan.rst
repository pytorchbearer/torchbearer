GAN Guide
====================================

We shall try to implement something more complicated using torchbearer - a Generative Adverserial Network (GAN).
This tutorial is a modified version of the GAN implimentation from the brilliant collection of GAN implementations here XXX by XXX.

Data and Constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first define all constants for the example

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 19-26

We then define the dataset and dataloader - for this example, MNIST.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 121-128

Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the generator and discriminator from XXX and combine them into a model that performs a single forward pass.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 75-91

Note that we have to be careful to remove the gradient information from the discriminator after doing the generator forward pass.

Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since our loss is complicated in this example, we shall forgo the basic loss criterion used in normal torchbearer models.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 159-160

Instead use a callback to provide the loss.
We also utilise this callback to add constants to state.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 94-108

Note that we have summed the separate discriminator and generator losses since their graphs are separated, this is allowable.

Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We would like to follow the discriminator and generator losses during training - note that we added these to state during the model definition.
We can then create metrics from these by decorating simple state fetcher metrics.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 137-145

Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can then train the torchbearer model in the standard way

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 162-164

Visualising
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We borrow the image saving method from the tutorial XXX and put it in a call back to save on training step.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 111-116

After 172400 iterations we see the following.

.. figure:: /_static/img/172400.png
   :scale: 200 %
   :alt: GAN generated samples after 172400 iterations


Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for the example is given below:

 :download:`Download Python source code: gan.py </_static/examples/gan.py>`



