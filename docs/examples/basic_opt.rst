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
   :lines: 7-27

The Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For function minimisation we have an analogue to neural network losses - we minimise the value of the function under the current estimates of the minimum.
Note that as we are using a base loss, torchbearer passes this the network output and the "label" (which is of no use here).

.. literalinclude:: /_static/examples/basic_opt.py
   :language: python
   :lines: 30-31


Optimising
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We need two more things before we can start optimising with torchbearer.
We need out initial guess - which we've set to [2.0, 1.0, 10.0] and we need to tell torchbearer how "long" an epoch is - ie how many optimisation steps we want for each epoch.
For our simple function, we can complete the optimisation in a single epoch, but for more complex optimisations we might want to take multiple epochs and include tensorboard logging and perhaps learning rate annealing to find a final solution.
We have set the number of optimisation steps for this example as 50000.

.. literalinclude:: /_static/examples/basic_opt.py
   :language: python
   :lines: 43-44

The learning rate chosen for this example is very low and we could get convergence much faster with a larger rate, however this allows us to view convergence in real time.
We define the model and optimiser in the standard way.

.. literalinclude:: /_static/examples/basic_opt.py
   :language: python
   :lines: 46-47

Finally we start the optimising (giving as "data" and "targets" the number of steps desired) and print the final minimum estimate.

.. literalinclude:: /_static/examples/basic_opt.py
   :language: python
   :lines: 49-51

Note that we could use targets that are meaningful as they are given to the loss function, however this is not done for this example.


Viewing Progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You might have noticed in the previous snippet that the example uses a metric we've not seen before.
This simple metric is used to display the estimate throughout the optimisation process - although this is probably only useful for very small optimisation problems.

.. literalinclude:: /_static/examples/basic_opt.py
   :language: python
   :lines: 34-40

The final estimate is very close to our desired minimum at [5, 0, 1]:

tensor([ 4.9988e+00,  4.5355e-05,  1.0004e+00])
