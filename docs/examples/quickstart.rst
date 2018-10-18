Quickstart Guide
====================================

This guide will give a quick intro to training PyTorch models with torchbearer. We'll start by loading in some data and defining a model, then we'll train it for a few epochs and see how well it does.

Defining the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's get using torchbearer. Here's some data from Cifar10 and a simple 3 layer strided CNN:

.. literalinclude:: /_static/examples/quickstart.py
   :language: python
   :lines: 9-52

Note that we use torchbearers :class:`.DatasetValidationSplitter` here to create a validation set (10% of the data).
This is essential to avoid `over-fitting to your test data <http://blog.kaggle.com/2012/07/06/the-dangers-of-overfitting-psychopathy-post-mortem/>`_.

Training on Cifar10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically we would need a training loop and a series of calls to backward, step etc.
Instead, with torchbearer, we can define our optimiser and some metrics (just 'acc' and 'loss' for now) and let it do the work.

.. literalinclude:: /_static/examples/quickstart.py
   :lines: 54-64

Running the above produces the following output:

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

Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for the example is given below:

 :download:`Download Python source code: quickstart.py </_static/examples/quickstart.py>`

