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
   0/10(t): 100%|██████████| 352/352 [00:02<00:00, 163.98it/s, acc=0.4339, loss=1.5776, running_acc=0.5202, running_loss=1.3494]
   0/10(v): 100%|██████████| 40/40 [00:00<00:00, 365.42it/s, val_acc=0.5266, val_loss=1.3208]
   1/10(t): 100%|██████████| 352/352 [00:02<00:00, 171.36it/s, acc=0.5636, loss=1.2176, running_acc=0.5922, running_loss=1.1418]
   1/10(v): 100%|██████████| 40/40 [00:00<00:00, 292.15it/s, val_acc=0.5888, val_loss=1.1657]
   2/10(t): 100%|██████████| 352/352 [00:02<00:00, 124.04it/s, acc=0.6226, loss=1.0671, running_acc=0.6222, running_loss=1.0566]
   2/10(v): 100%|██████████| 40/40 [00:00<00:00, 359.21it/s, val_acc=0.626, val_loss=1.0555]
   3/10(t): 100%|██████████| 352/352 [00:02<00:00, 151.69it/s, acc=0.6587, loss=0.972, running_acc=0.6634, running_loss=0.9589]
   3/10(v): 100%|██████████| 40/40 [00:00<00:00, 222.62it/s, val_acc=0.6414, val_loss=1.0064]
   4/10(t): 100%|██████████| 352/352 [00:02<00:00, 131.49it/s, acc=0.6829, loss=0.9061, running_acc=0.6764, running_loss=0.918]
   4/10(v): 100%|██████████| 40/40 [00:00<00:00, 346.88it/s, val_acc=0.6636, val_loss=0.9449]
   5/10(t): 100%|██████████| 352/352 [00:02<00:00, 164.28it/s, acc=0.6988, loss=0.8563, running_acc=0.6919, running_loss=0.858]
   5/10(v): 100%|██████████| 40/40 [00:00<00:00, 244.97it/s, val_acc=0.663, val_loss=0.9404]
   6/10(t): 100%|██████████| 352/352 [00:02<00:00, 149.52it/s, acc=0.7169, loss=0.8131, running_acc=0.7095, running_loss=0.8421]
   6/10(v): 100%|██████████| 40/40 [00:00<00:00, 329.26it/s, val_acc=0.6704, val_loss=0.9209]
   7/10(t): 100%|██████████| 352/352 [00:02<00:00, 160.60it/s, acc=0.7302, loss=0.7756, running_acc=0.738, running_loss=0.767]
   7/10(v): 100%|██████████| 40/40 [00:00<00:00, 349.86it/s, val_acc=0.6716, val_loss=0.9313]
   8/10(t): 100%|██████████| 352/352 [00:02<00:00, 155.08it/s, acc=0.7412, loss=0.7444, running_acc=0.7347, running_loss=0.7547]
   8/10(v): 100%|██████████| 40/40 [00:00<00:00, 350.05it/s, val_acc=0.673, val_loss=0.9324]
   9/10(t): 100%|██████████| 352/352 [00:02<00:00, 165.28it/s, acc=0.7515, loss=0.715, running_acc=0.7352, running_loss=0.7492]
   9/10(v): 100%|██████████| 40/40 [00:00<00:00, 310.76it/s, val_acc=0.6792, val_loss=0.9743]
   0/1(e): 100%|██████████| 79/79 [00:00<00:00, 233.06it/s, test_acc=0.6673, test_loss=0.9741]

Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for the example is given below:

 :download:`Download Python source code: quickstart.py </_static/examples/quickstart.py>`

