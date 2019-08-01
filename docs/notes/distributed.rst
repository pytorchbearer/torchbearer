Using DistributedDataParallel with Torchbearer on CPU
=====================================================

This note will quickly cover how we can use torchbearer to train over multiple nodes.
We shall do this by training a simple model to classify and for a massive amount of overkill we will be doing this on MNIST.
Most of the code for this example is based off the
`Distributed Data Parallel (DDP) tutorial <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`__ and the
`imagenet example <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`__
from the PyTorch docs.
We recommend you read at least the DDP tutorial before continuing with this note.

Setup, Cleanup and Model
------------------------------------
We keep similar setup, cleanup and model from the DDP tutorial. All that is changed is taking rank, world size and master
address from terminal arguments and changing the model to apply to MNIST.
Note that we are keeping to the GLOO backend since this part of the note will be purely on the CPU.

.. literalinclude:: /_static/examples/distributed_data_parallel.py
   :lines: 23-48



Sync Methods
------------------------------------
Since we are working across multiple machines we need a way to synchronise the model itself and its gradients. To do this
we utilise methods similar to that of the `distributed applications tutorial <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`__
from PyTorch.

.. literalinclude:: /_static/examples/distributed_data_parallel.py
   :lines: 51-62

Since we require the gradients to be synced every step we implement both of these methods as Torchbearer callbacks.
We sync the model itself on init and sync the gradients every step after the backward call.

.. literalinclude:: /_static/examples/distributed_data_parallel.py
   :lines: 65-72


Worker Function
------------------------------------
Now we need to define the main worker function that each process will be running. We need this to setup the environment,
actually run the training process and cleanup the environment after we finish.
This function outside of calling `setup` and `cleanup` is exactly the same as any Torchbearer training function.

.. literalinclude:: /_static/examples/distributed_data_parallel.py
   :lines: 80-119

You might have noticed that we had an extra flatten callback in the Trial, the only purpose of this was to flatten each image.

.. literalinclude:: /_static/examples/distributed_data_parallel.py
   :lines: 75-77


Running
------------------------------------
All we need to do now is write a `__main__` function to run the worker function.

.. literalinclude:: /_static/examples/distributed_data_parallel.py
   :lines: 122-124

We can then ssh into each node on which we want to run the training and run the following code replacing i with the rank of each process.

.. highlight:: bash

.. code:: bash

    python distributed_data_parallel.py --world-size 2 --rank i --host (host address)

Running on machines with GPUs
------------------------------------
Coming soon.


Source Code
------------------------------------

The source code for this example is given below:

 :download:`Download Python source code: distributed_data_parallel.py </_static/examples/distributed_data_parallel.py>`
