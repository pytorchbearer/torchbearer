Quickstart Guide
====================================

This guide will give a quick intro to training PyTorch models with torchbearer. We'll start by loading in some data and defining a model, then we'll train it for a few epochs and see how well it does.

Defining the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's get using torchbearer. Here's some data from Cifar10 and a simple 3 layer strided CNN:

.. literalinclude:: /_static/examples/quickstart.py
   :language: python
   :lines: 7-45

Typically we would need a training loop and a series of calls to backward, step etc.
Instead, with torchbearer, we can define our optimiser and some metrics (just 'acc' and 'loss' for now) and let it do the work.

Training on Cifar10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: /_static/examples/quickstart.py
   :lines: 47-53

Running the above produces the following output:

.. code::

    Files already downloaded and verified
    Files already downloaded and verified
    0/10(t): 100%|██████████| 391/391 [00:01<00:00, 211.19it/s, running_acc=0.549, running_loss=1.25, acc=0.469, acc_std=0.499, loss=1.48, loss_std=0.238]
    0/10(v): 100%|██████████| 79/79 [00:00<00:00, 265.14it/s, val_acc=0.556, val_acc_std=0.497, val_loss=1.25, val_loss_std=0.0785]
    1/10(t): 100%|██████████| 391/391 [00:01<00:00, 209.80it/s, running_acc=0.61, running_loss=1.09, acc=0.593, acc_std=0.491, loss=1.14, loss_std=0.0968]
    1/10(v): 100%|██████████| 79/79 [00:00<00:00, 227.97it/s, val_acc=0.593, val_acc_std=0.491, val_loss=1.14, val_loss_std=0.0865]
    2/10(t): 100%|██████████| 391/391 [00:01<00:00, 220.70it/s, running_acc=0.656, running_loss=0.972, acc=0.645, acc_std=0.478, loss=1.01, loss_std=0.0915]
    2/10(v): 100%|██████████| 79/79 [00:00<00:00, 218.91it/s, val_acc=0.631, val_acc_std=0.482, val_loss=1.04, val_loss_std=0.0951]
    3/10(t): 100%|██████████| 391/391 [00:01<00:00, 208.67it/s, running_acc=0.682, running_loss=0.906, acc=0.675, acc_std=0.468, loss=0.922, loss_std=0.0895]
    3/10(v): 100%|██████████| 79/79 [00:00<00:00, 86.95it/s, val_acc=0.657, val_acc_std=0.475, val_loss=0.97, val_loss_std=0.0925]
    4/10(t): 100%|██████████| 391/391 [00:01<00:00, 211.22it/s, running_acc=0.693, running_loss=0.866, acc=0.699, acc_std=0.459, loss=0.86, loss_std=0.092]
    4/10(v): 100%|██████████| 79/79 [00:00<00:00, 249.74it/s, val_acc=0.662, val_acc_std=0.473, val_loss=0.957, val_loss_std=0.093]
    5/10(t): 100%|██████████| 391/391 [00:01<00:00, 205.12it/s, running_acc=0.71, running_loss=0.826, acc=0.713, acc_std=0.452, loss=0.818, loss_std=0.0904]
    5/10(v): 100%|██████████| 79/79 [00:00<00:00, 230.12it/s, val_acc=0.661, val_acc_std=0.473, val_loss=0.962, val_loss_std=0.0966]
    6/10(t): 100%|██████████| 391/391 [00:01<00:00, 210.87it/s, running_acc=0.714, running_loss=0.81, acc=0.727, acc_std=0.445, loss=0.779, loss_std=0.0904]
    6/10(v): 100%|██████████| 79/79 [00:00<00:00, 241.95it/s, val_acc=0.667, val_acc_std=0.471, val_loss=0.952, val_loss_std=0.11]
    7/10(t): 100%|██████████| 391/391 [00:01<00:00, 209.94it/s, running_acc=0.727, running_loss=0.791, acc=0.74, acc_std=0.439, loss=0.747, loss_std=0.0911]
    7/10(v): 100%|██████████| 79/79 [00:00<00:00, 223.23it/s, val_acc=0.673, val_acc_std=0.469, val_loss=0.938, val_loss_std=0.107]
    8/10(t): 100%|██████████| 391/391 [00:01<00:00, 203.16it/s, running_acc=0.747, running_loss=0.736, acc=0.752, acc_std=0.432, loss=0.716, loss_std=0.0899]
    8/10(v): 100%|██████████| 79/79 [00:00<00:00, 221.55it/s, val_acc=0.679, val_acc_std=0.467, val_loss=0.923, val_loss_std=0.113]
    9/10(t): 100%|██████████| 391/391 [00:01<00:00, 213.23it/s, running_acc=0.756, running_loss=0.701, acc=0.759, acc_std=0.428, loss=0.695, loss_std=0.0915]
    9/10(v): 100%|██████████| 79/79 [00:00<00:00, 245.33it/s, val_acc=0.676, val_acc_std=0.468, val_loss=0.951, val_loss_std=0.111]

Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for the example is given below:

 :download:`Download Python source code: quickstart.py </_static/examples/quickstart.py>`

