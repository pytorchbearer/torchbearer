Training a Variational Auto-Encoder
====================================

This guide will give a quick guide on training a variational auto-encoder (VAE) in sconce. We will use the VAE example from the pytorch examples here_:

.. _here: https://github.com/pytorch/examples/tree/master/vae

Defining the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We shall first copy the VAE example model.

.. literalinclude:: ../../examples/vae_standard.py
   :language: python
   :lines: 44-73

Defining the Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First we get the MNIST dataset from torchvision and transform them to torch tensors.

.. literalinclude:: ../../examples/vae_standard.py
   :language: python
   :lines: 23-31

The output label from this dataset is the classification label, since we are doing a auto-encoding problem, we wish the label to be the original image. To fix this we create a wrapper class to replace the classification label with the image.

.. literalinclude:: ../../examples/vae_standard.py
   :language: python
   :lines: 10-20

We then wrap the original datasets and create training and testing data generators in the standard pytorch way.

.. literalinclude:: ../../examples/vae_standard.py
   :language: python
   :lines: 33-41

Defining the Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we have the model, we will need a loss function to optimize. VAEs typically use the sum of a reconstruction loss and a KL-divergence loss.

.. literalinclude:: ../../examples/vae_standard.py
   :language: python
   :lines: 82-87

This requires the packing of the reconstruction, mean and log-variance into the model output and unpacking it for the loss function to use

.. literalinclude:: ../../examples/vae_standard.py
   :language: python
   :lines: 70-73

.. literalinclude:: ../../examples/vae_standard.py
   :language: python
   :lines: 76-79

However, in sconce, there is a persistent state dictionary which can be used to conveniently hold intermediate tensors such as the mean and log-variance.

By default the model forward pass does not have access to the state dictionary, but setting the `pass_state` flag to true in the fit_generator_ call gives the model access to state on forward.

.. _fit_generator: https://pysconce.readthedocs.io/en/latest/code/main.html#sconce.sconce.Model.fit_generator

.. literalinclude:: ../../examples/vae.py
   :language: python
   :lines: 120

We can then modify the model forward pass to store the mean and log-variance under suitable keys.

.. literalinclude:: ../../examples/vae.py
   :language: python
   :lines: 74-79

The loss can then be separated into a standard reconstruction loss and a separate KL-divergence loss using intermediate tensor values.

.. literalinclude:: ../../examples/vae.py
   :language: python
   :lines: 82-84

Since loss functions cannot access state, we can utilise a simple callback to complete the loss calculation.

.. literalinclude:: ../../examples/vae.py
   :language: python
   :lines: 87-95


Visualising Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For auto-encoding problems it is often useful to visualise the reconstructions. We can do this in sconce by using another simple callback. We stack the first 8 images from the first validation batch and pass them to torchvisions_ save_image_ function which saves out visualisations.

.. _torchvisions: https://github.com/pytorch/vision
.. _save_image: https://pytorch.org/docs/stable/torchvision/utils.html?highlight=save#torchvision.utils.save_image

.. literalinclude:: ../../examples/vae.py
   :lines: 98-112

Training the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We train the model by creating a torchmodel and a sconcemodel and calling fit_generator.

.. literalinclude:: ../../examples/vae.py
   :lines: 115-120

The visualised results after ten epochs then look like this:

.. figure:: /_static/img/reconstruction_9.png
   :scale: 200 %
   :alt: VAE reconstructions after 10 epochs of mnist
