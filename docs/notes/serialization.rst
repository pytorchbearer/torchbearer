Serializing a Trial
====================================

This guide will explain the two different ways to how to save and reload your results from a Trial.

Setting up a Mock Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's assume we have a basic binary classification task where we have 100-dimensional samples as input and a binary label as output.
Let's also assume that we would like to solve this problem with a 2-layer neural network.
Finally, we also want to keep track of the sum of hidden outputs for some arbitrary reason. Therefore we use the state functionality of Torchbearer.

We create a state key for the mock sum we wanted to track using state.

.. literalinclude:: /_static/examples/serialization.py
   :language: python
   :lines: 9

Here is our basic 2-layer neural network.

.. literalinclude:: /_static/examples/serialization.py
   :language: python
   :lines: 12-23

We create some random training dataset and put them in a DataLoader.

.. literalinclude:: /_static/examples/serialization.py
   :language: python
   :lines: 27-30

Let's say we would like to save the model every time we get a better training loss. Torchbearer's Best checkpoint callback is perfect for this job.
We then run the model for 3 epochs.

.. literalinclude:: /_static/examples/serialization.py
   :language: python
   :lines: 33-40

Reloading the Trial for More Epochs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given we recreate the exact same Trial structure, we can easily resume our run from the last checkpoint. The following code block shows how it's done.
Remember here that the ``epochs`` parameter we pass to Trial acts cumulative. In other words, the following run will complement the entire training to
a total of 6 epochs.

.. literalinclude:: /_static/examples/serialization.py
   :language: python
   :lines: 43-49

Trying to Reload to a PyTorch Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We try to load the ``state_dict`` to a regular PyTorch Module, as described in PyTorch's own documentation here_:

.. _here: https://pytorch.org/docs/stable/notes/serialization.html

.. literalinclude:: /_static/examples/serialization.py
   :language: python
   :lines: 52-57

We will get the following error:

.. code::

    'StateKey' object has no attribute 'startswith'

The reason is that the ``state_dict`` has Trial related attributes that are unknown to a native PyTorch model. This is why we have the ``save_model_params_only``
option for our checkpointers. We try again with that option

.. literalinclude:: /_static/examples/serialization.py
   :language: python
   :lines: 61-72

No errors this time, but we still have to test. Here is a test sample and we run it through the model.

.. literalinclude:: /_static/examples/serialization.py
   :language: python
   :lines: 73-78

.. code::

    forward() missing 1 required positional argument: 'state'

Now we get a different error, stating that we should also be passing ``state`` as an argument to module's forward. This should not be a surprise
as we defined ``state`` parameter in the forward method of ``BasicModule`` as a required argument.

Robust Signature for Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We define the model with a better signature this time, so it gracefully handles the problem above.

.. literalinclude:: /_static/examples/serialization.py
   :language: python
   :lines: 81-94

Finally, we wrap it up once again to test the new definition of the model.


.. literalinclude:: /_static/examples/serialization.py
   :language: python
   :lines: 98-111

Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for the example are given below:

 :download:`Download Python source code: serialization.py </_static/examples/serialization.py>`