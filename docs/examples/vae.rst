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

We train the model by creating a torchmodel and a torchbearertrialand calling run_. As our loss is named binary_cross_entropy, we can use the 'acc' metric to get a binary accuracy.

.. _run: https://torchbearer.readthedocs.io/en/latest/code/main.html#torchbearer.trial.Trial.run


.. literalinclude:: /_static/examples/vae.py
   :lines: 118-127

This gives the following output:

.. code::

    0/10(t): 100%|██████████| 422/422 [00:01<00:00, 219.71it/s, binary_acc=0.9139, loss=2.139e+4, loss_std=6582, running_binary_acc=0.9416, running_loss=1.685e+4]
    0/10(v): 100%|██████████| 47/47 [00:00<00:00, 269.77it/s, val_binary_acc=0.9505, val_loss=1.558e+4, val_loss_std=470.8]
    1/10(t): 100%|██████████| 422/422 [00:01<00:00, 219.80it/s, binary_acc=0.9492, loss=1.573e+4, loss_std=573.6, running_binary_acc=0.9531, running_loss=1.52e+4]
    1/10(v): 100%|██████████| 47/47 [00:00<00:00, 300.54it/s, val_binary_acc=0.9614, val_loss=1.399e+4, val_loss_std=427.7]
    2/10(t): 100%|██████████| 422/422 [00:01<00:00, 232.41it/s, binary_acc=0.9558, loss=1.476e+4, loss_std=407.3, running_binary_acc=0.9571, running_loss=1.457e+4]
    2/10(v): 100%|██████████| 47/47 [00:00<00:00, 296.49it/s, val_binary_acc=0.9652, val_loss=1.336e+4, val_loss_std=338.2]
    3/10(t): 100%|██████████| 422/422 [00:01<00:00, 213.10it/s, binary_acc=0.9585, loss=1.437e+4, loss_std=339.6, running_binary_acc=0.9595, running_loss=1.423e+4]
    3/10(v): 100%|██████████| 47/47 [00:00<00:00, 313.42it/s, val_binary_acc=0.9672, val_loss=1.304e+4, val_loss_std=372.3]
    4/10(t): 100%|██████████| 422/422 [00:01<00:00, 213.43it/s, binary_acc=0.9601, loss=1.413e+4, loss_std=332.5, running_binary_acc=0.9605, running_loss=1.409e+4]
    4/10(v): 100%|██████████| 47/47 [00:00<00:00, 242.23it/s, val_binary_acc=0.9683, val_loss=1.282e+4, val_loss_std=369.3]
    5/10(t): 100%|██████████| 422/422 [00:01<00:00, 220.94it/s, binary_acc=0.9611, loss=1.398e+4, loss_std=300.9, running_binary_acc=0.9614, running_loss=1.397e+4]
    5/10(v): 100%|██████████| 47/47 [00:00<00:00, 316.69it/s, val_binary_acc=0.9689, val_loss=1.281e+4, val_loss_std=423.6]
    6/10(t): 100%|██████████| 422/422 [00:01<00:00, 230.53it/s, binary_acc=0.9619, loss=1.385e+4, loss_std=292.1, running_binary_acc=0.9621, running_loss=1.38e+4]
    6/10(v): 100%|██████████| 47/47 [00:00<00:00, 241.06it/s, val_binary_acc=0.9695, val_loss=1.275e+4, val_loss_std=459.9]
    7/10(t): 100%|██████████| 422/422 [00:01<00:00, 227.49it/s, binary_acc=0.9624, loss=1.377e+4, loss_std=306.9, running_binary_acc=0.9624, running_loss=1.381e+4]
    7/10(v): 100%|██████████| 47/47 [00:00<00:00, 237.75it/s, val_binary_acc=0.97, val_loss=1.258e+4, val_loss_std=353.8]
    8/10(t): 100%|██████████| 422/422 [00:01<00:00, 220.68it/s, binary_acc=0.9629, loss=1.37e+4, loss_std=300.8, running_binary_acc=0.9629, running_loss=1.369e+4]
    8/10(v): 100%|██████████| 47/47 [00:00<00:00, 301.59it/s, val_binary_acc=0.9704, val_loss=1.255e+4, val_loss_std=347.7]
    9/10(t): 100%|██████████| 422/422 [00:01<00:00, 215.23it/s, binary_acc=0.9633, loss=1.364e+4, loss_std=310, running_binary_acc=0.9633, running_loss=1.366e+4]
    9/10(v): 100%|██████████| 47/47 [00:00<00:00, 309.51it/s, val_binary_acc=0.9707, val_loss=1.25e+4, val_loss_std=358.9]

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