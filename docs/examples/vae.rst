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

   Files already downloaded and verified
   Files already downloaded and verified
   0/10(t): 100%|██████████| 352/352 [00:01<00:00, 233.36it/s, running_acc=0.536, running_loss=1.32, acc=0.459, acc_std=0.498, loss=1.52, loss_std=0.239]
   0/10(v): 100%|██████████| 40/40 [00:00<00:00, 239.40it/s, val_acc=0.536, val_acc_std=0.499, val_loss=1.29, val_loss_std=0.0731]
   1/10(t): 100%|██████████| 352/352 [00:01<00:00, 211.19it/s, running_acc=0.599, running_loss=1.13, acc=0.578, acc_std=0.494, loss=1.18, loss_std=0.096]
   1/10(v): 100%|██████████| 40/40 [00:00<00:00, 232.97it/s, val_acc=0.594, val_acc_std=0.491, val_loss=1.14, val_loss_std=0.101]
   2/10(t): 100%|██████████| 352/352 [00:01<00:00, 216.68it/s, running_acc=0.636, running_loss=1.04, acc=0.631, acc_std=0.482, loss=1.04, loss_std=0.0944]
   2/10(v): 100%|██████████| 40/40 [00:00<00:00, 210.73it/s, val_acc=0.626, val_acc_std=0.484, val_loss=1.07, val_loss_std=0.0974]
   3/10(t): 100%|██████████| 352/352 [00:01<00:00, 190.88it/s, running_acc=0.671, running_loss=0.929, acc=0.664, acc_std=0.472, loss=0.957, loss_std=0.0929]
   3/10(v): 100%|██████████| 40/40 [00:00<00:00, 221.79it/s, val_acc=0.639, val_acc_std=0.48, val_loss=1.02, val_loss_std=0.103]
   4/10(t): 100%|██████████| 352/352 [00:01<00:00, 212.43it/s, running_acc=0.685, running_loss=0.897, acc=0.689, acc_std=0.463, loss=0.891, loss_std=0.0888]
   4/10(v): 100%|██████████| 40/40 [00:00<00:00, 249.99it/s, val_acc=0.655, val_acc_std=0.475, val_loss=0.983, val_loss_std=0.113]
   5/10(t): 100%|██████████| 352/352 [00:01<00:00, 209.45it/s, running_acc=0.711, running_loss=0.835, acc=0.706, acc_std=0.456, loss=0.844, loss_std=0.088]
   5/10(v): 100%|██████████| 40/40 [00:00<00:00, 240.80it/s, val_acc=0.648, val_acc_std=0.477, val_loss=0.965, val_loss_std=0.107]
   6/10(t): 100%|██████████| 352/352 [00:01<00:00, 216.89it/s, running_acc=0.713, running_loss=0.826, acc=0.72, acc_std=0.449, loss=0.802, loss_std=0.0903]
   6/10(v): 100%|██████████| 40/40 [00:00<00:00, 238.17it/s, val_acc=0.655, val_acc_std=0.475, val_loss=0.97, val_loss_std=0.0997]
   7/10(t): 100%|██████████| 352/352 [00:01<00:00, 213.82it/s, running_acc=0.737, running_loss=0.773, acc=0.734, acc_std=0.442, loss=0.765, loss_std=0.0878]
   7/10(v): 100%|██████████| 40/40 [00:00<00:00, 202.45it/s, val_acc=0.677, val_acc_std=0.468, val_loss=0.936, val_loss_std=0.0985]
   8/10(t): 100%|██████████| 352/352 [00:01<00:00, 211.36it/s, running_acc=0.732, running_loss=0.744, acc=0.746, acc_std=0.435, loss=0.728, loss_std=0.0902]
   8/10(v): 100%|██████████| 40/40 [00:00<00:00, 204.52it/s, val_acc=0.674, val_acc_std=0.469, val_loss=0.949, val_loss_std=0.124]
   9/10(t): 100%|██████████| 352/352 [00:02<00:00, 171.22it/s, running_acc=0.738, running_loss=0.737, acc=0.749, acc_std=0.434, loss=0.723, loss_std=0.0885]
   9/10(v): 100%|██████████| 40/40 [00:00<00:00, 188.51it/s, val_acc=0.669, val_acc_std=0.471, val_loss=0.97, val_loss_std=0.173]
   0/1(e): 100%|██████████| 79/79 [00:00<00:00, 241.00it/s, test_acc=0.675, test_acc_std=0.468, test_loss=0.952, test_loss_std=0.109]

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