Optimising functions
====================================

Now for something a bit different.
PyTorch is a tensor processing library and whilst it has a focus on neural networks, it can also be used for more standard funciton optimisation.
In this example we will use torchbearer to minimise a simple function.


The Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First we will need to create something that looks very similar to a neural network model - but with the purpose of minimising our function.
We store the current estimates for the minimum as parameters in the model (so PyTorch optimisers can find and optimise them) and we return the function value in the forward method.

.. literalinclude:: /_static/examples/basic_opt.py
   :language: python
   :lines: 9-29

The Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For function minimisation we have an analogue to neural network losses - we minimise the value of the function under the current estimates of the minimum.
Note that as we are using a base loss, torchbearer passes this the network output and the "label" (which is of no use here).

.. literalinclude:: /_static/examples/basic_opt.py
   :language: python
   :lines: 32-33


Optimising
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We need two more things before we can start optimising with torchbearer.
We need our initial guess - which we've set to [2.0, 1.0, 10.0] and we need to tell torchbearer how "long" an epoch is - I.e. how many optimisation steps we want for each epoch.
For our simple function, we can complete the optimisation in a single epoch, but for more complex optimisations we might want to take multiple epochs and include tensorboard logging and perhaps learning rate annealing to find a final solution.
We have set the number of optimisation steps for this example as 50000.

.. literalinclude:: /_static/examples/basic_opt.py
   :language: python
   :lines: 36-37

The learning rate chosen for this example is very low and we could get convergence much faster with a larger rate, however this allows us to view convergence in real time.
We define the model and optimiser in the standard way.

.. literalinclude:: /_static/examples/basic_opt.py
   :language: python
   :lines: 39-40

Finally we start the optimising on the GPU and print the final minimum estimate.

.. literalinclude:: /_static/examples/basic_opt.py
   :language: python
   :lines: 42-45

Usually torchbearer will infer the number of training steps from the data generator.
Since for this example we have no data to give the model (which will be passed `None`), we need to tell torchbearer how many steps to run using the ``for_train_steps`` method.


Viewing Progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You might have noticed in the previous snippet that the example uses a metric we've not seen before.
The state key that represents our estimate in state can also act as a metric and is created at the beginning of the file with:

.. literalinclude:: /_static/examples/basic_opt.py
   :language: python
   :lines: 8

Putting all of it together and running provides the following output:

.. code::

    0/1(t): 100%|██████████| 50000/50000 [00:54<00:00, 912.37it/s, est=[4.9988 0.     1.0004], running_loss=1.6e-06, loss=4.55, loss_std=13.7]

The final estimate is very close to the true minimum at [5, 0, 1]:

tensor([ 4.9988e+00,  4.5355e-05,  1.0004e+00])

Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for the example is given below:

 :download:`Download Python source code: basic_opt.py </_static/examples/basic_opt.py>`
