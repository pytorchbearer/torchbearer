Training a Variational Auto-Encoder
====================================

This guide will give a quick guide on training a variational auto-encoder (VAE) in torchbearer. We will use the VAE example from the pytorch examples here_:

.. _here: https://github.com/pytorch/examples/tree/master/vae

Defining the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We shall first copy the VAE example model.

.. literalinclude:: /_static/examples/vae_standard.py
   :language: python
   :lines: 46-75

Defining the Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We get the MNIST dataset from torchvision, split it into a train and validation set and transform them to torch tensors.

.. literalinclude:: /_static/examples/vae_standard.py
   :language: python
   :lines: 24-33

The output label from this dataset is the classification label, since we are doing a auto-encoding problem, we wish the label to be the original image. To fix this we create a wrapper class which replaces the classification label with the image.

.. literalinclude:: /_static/examples/vae_standard.py
   :language: python
   :lines: 11-21

We then wrap the original datasets and create training and testing data generators in the standard pytorch way.

.. literalinclude:: /_static/examples/vae_standard.py
   :language: python
   :lines: 37-43

Defining the Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now we have the model and data, we will need a loss function to optimize.
VAEs typically take the sum of a reconstruction loss and a KL-divergence loss to form the final loss value.

.. literalinclude:: /_static/examples/vae.py
   :language: python
   :lines: 86-88

.. literalinclude:: /_static/examples/vae.py
   :language: python
   :lines: 91-93

There are two ways this can be done in torchbearer - one is very similar to the PyTorch example method and the other utilises the torchbearer state.

PyTorch method
------------------------------------

The loss function slightly modified from the PyTorch example is:

.. literalinclude:: /_static/examples/vae_standard.py
   :language: python
   :lines: 89-97

This requires the packing of the reconstruction, mean and log-variance into the model output and unpacking it for the loss function to use.

.. literalinclude:: /_static/examples/vae_standard.py
   :language: python
   :lines: 72-75


Using Torchbearer State
------------------------------------

Instead of having to pack and unpack the mean and variance in the forward pass, in torchbearer there is a persistent state dictionary which can be used to conveniently hold such intermediate tensors.
We can (and should) generate unique state keys for interacting with state:

.. literalinclude:: /_static/examples/vae.py
   :language: python
   :lines: 48-49


By default the model forward pass does not have access to the state dictionary, but setting the ``pass_state`` flag to true when initialising Trial_ gives the model access to state on forward.

.. _Trial: https://torchbearer.readthedocs.io/en/latest/code/main.html#torchbearer.trial.Trial

.. literalinclude:: /_static/examples/vae.py
   :language: python
   :lines: 122-125

We can then modify the model forward pass to store the mean and log-variance under suitable keys.

.. literalinclude:: /_static/examples/vae.py
   :language: python
   :lines: 78-83

The reconstruction loss is a standard loss taking network output and the true label

.. literalinclude:: /_static/examples/vae.py
   :language: python
   :lines: 120

Since loss functions cannot access state, we utilise a simple callback to combine the kld loss which does not act on network output or true label.

.. literalinclude:: /_static/examples/vae.py
   :language: python
   :lines: 96-99


Visualising Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For auto-encoding problems it is often useful to visualise the reconstructions. We can do this in torchbearer by using another simple callback. We stack the first 8 images from the first validation batch and pass them to torchvisions_ save_image_ function which saves out visualisations.

.. _torchvisions: https://github.com/pytorch/vision
.. _save_image: https://pytorch.org/docs/stable/torchvision/utils.html?highlight=save#torchvision.utils.save_image

.. literalinclude:: /_static/examples/vae.py
   :lines: 102-115

Training the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We train the model by creating a torchmodel and a torchbearertrialand calling run_.

.. _run: https://torchbearer.readthedocs.io/en/latest/code/main.html#torchbearer.trial.Trial.run


.. literalinclude:: /_static/examples/vae.py
   :lines: 118-127

The visualised results after ten epochs then look like this:

.. figure:: /_static/img/reconstruction_9.png
   :scale: 200 %
   :alt: VAE reconstructions after 10 epochs of mnist

Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for the example are given below:

Standard:

 :download:`Download Python source code: vae_standard.py </_static/examples/vae_standard.py>`

Using state:

 :download:`Download Python source code: vae.py </_static/examples/vae.py>`