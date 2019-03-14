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
   :lines: 20-32

We then define a number of state keys for convenience using :func:`.state_key`. This is optional, however, it automatically avoids key conflicts.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 35-47

We then define the dataset and dataloader - for this example, MNIST.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 120-125

Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the generator and discriminator from PyTorch_GAN_.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 50-94

We then create the models and optimisers.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 128-132

Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GANs usually require two different losses, one for the generator and one for the discriminator.
We define these as functions of state so that we can access the discriminator model for the additional forward passes required.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 97-108

We will see later how we get a torchbearer trial to use these losses.

Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We would like to follow the discriminator and generator losses during training - note that we added these to state during the model definition.
In torchbearer, state keys are also metrics, so we can take means and running means of them and tell torchbearer to output them as metrics.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 145-146

We will add this metric list to the trial when we create it.


Closures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The training loop of a GAN is a bit different to a standard model training loop.
GANs require separate forward and backward passes for the generator and discriminator.
To achieve this in torchbearer we can write a new closure.
Since the individual training loops for the generator and discriminator are the same as a
standard training loop we can use a :func:`.base_closure`.
The base closure takes state keys for required objects (data, model, optimiser, etc.) and returns a standard closure consisting of:

1. Zero gradients
2. Forward pass
3. Loss calculation
4. Backward pass

We create a separate closure for the generator and discriminator. We use separate state keys for some objects so we can use them separately, although the loss is easier to deal with in a single key.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 15, 135-136

We then create a main closure (a simple function of state) that runs both of these and steps the optimisers.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 139-143

We will add this closure to the trial next.


Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now create the torchbearer trial on the GPU in the standard way.
Note that when torchbearer is passed a ``None`` optimiser it creates a mock optimser that will just run the closure.
Since we are using the standard torchbearer state keys for the generator model and criterion, we can pass them in here.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 148-150

We now update state with the keys required for the discriminators closure and add the new closure to the trial.
Note that torchbearer doesn't know the discriminator model is a model here, so we have to sent it to the GPU ourselves.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 152-154

Finally we run the trial.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 155

Visualising
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We borrow the image saving method from PyTorch_GAN_ and put it in a call back to save :func:`~torchbearer.callbacks.decorators.on_step_training`.
We generate from the same inputs each time to get a better visualisation.

.. literalinclude:: /_static/examples/gan.py
   :language: python
   :lines: 111-115

Here is a Gif created from the saved images.

.. figure:: /_static/img/gan.gif
   :scale: 100 %
   :alt: GAN generated samples


Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for the example is given below:

 :download:`Download Python source code: gan.py </_static/examples/gan.py>`



