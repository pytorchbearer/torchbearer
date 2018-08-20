Using the Tensorboard Callback
====================================

In this note we will cover the use of the :class:`TensorBoard callback <.TensorBoard>`. This is one of three callbacks
in torchbearer which use the `TensorboardX <https://github.com/lanpa/tensorboardX>`_ library. The PyPi version of
tensorboardX (1.4) is somewhat outdated at the time of writing so it may be worth installing from source if some of the
examples don't run correctly:

.. code-block:: bash

    pip install git+https://github.com/lanpa/tensorboardX

The :class:`TensorBoard callback <.TensorBoard>` is simply used to log metric values (and optionally a model graph) to
tensorboard. Let's have a look at some examples.

Setup
------------------------------------

We'll begin with the data and simple model from our `quickstart example <../examples/quickstart.html>`_.

.. literalinclude:: /_static/examples/tensorboard.py
   :lines: 9-26

.. literalinclude:: /_static/examples/tensorboard.py
   :lines: 29-55

The callback has three capabilities that we will demonstrate in this guide:

#. It can log a graph of the model
#. It can log the batch metrics
#. It can log the epoch metrics

Logging the Model Graph
------------------------------------

One of the advantages of PyTorch is that it doesn't construct a model graph internally like other frameworks such as
TensorFlow. This means that determining the model structure requires a forward pass through the model with some dummy
data and parsing the subsequent graph built by autograd. Thankfully,
`TensorboardX <https://github.com/lanpa/tensorboardX>`_ can do this for us. The
:class:`TensorBoard callback <.TensorBoard>` makes things a little easier by creating the dummy data for us and handling
the interaction with `TensorboardX <https://github.com/lanpa/tensorboardX>`_. The size of the dummy data is chosen to
match the size of the data in the dataset / data loader, this means that we need at least one batch of training data for
the graph to be written. Let's train for one epoch just to see a model graph:

.. literalinclude:: /_static/examples/tensorboard.py
   :lines: 57-62

To see the result, navigate to the project directory and execute the command :code:`tensorboard --logdir logs`, then
open a web browser and navigate to `localhost:6006 <http://localhost:6006>`_. After a bit of clicking around you should
be able to see and download something like the following:

.. figure:: /_static/img/model_graph.png
   :scale: 25 %
   :alt: Simple model graph in tensorboard

The dynamic graph construction does introduce some weirdness, however, this is about as good as model graphs typically
get.

Logging Batch Metrics
------------------------------------

If we have some metrics that output every batch, we might want to log them to tensorboard. This is useful particularly
if epochs are long and we want to watch them progress. For this we can set :code:`write_batch_metrics=True` in the
:class:`TensorBoard callback <.TensorBoard>` constructor. Setting this flag will cause the batch metrics to be written
as graphs to tensorboard. We are also able to change the frequency of updates by choosing the :code:`batch_step_size`.
This is the number of batches to wait between updates and can help with reducing the size of the logs, 10 seems
reasonable. We run this for 10 epochs with the following:

.. literalinclude:: /_static/examples/tensorboard.py
   :lines: 64-66

Runnng tensorboard again with :code:`tensorboard --logdir logs`, navigating to
`localhost:6006 <http://localhost:6006>`_ and selecting 'WALL' for the horizontal axis we can see the following:

.. figure:: /_static/img/batch_metrics.png
   :scale: 25 %
   :alt: Batch metric graphs in tensorboard

Logging Epoch Metrics
------------------------------------

Logging epoch metrics is perhaps the most typical use case of TensorBoard and the
:class:`TensorBoard callback <.TensorBoard>`. Using the same model as before, but setting
:code:`write_epoch_metrics=True` we can log epoch metrics with the following:

.. literalinclude:: /_static/examples/tensorboard.py
   :lines: 68-70

Again, runnng tensorboard with :code:`tensorboard --logdir logs` and navigating to
`localhost:6006 <http://localhost:6006>`_ we see the following:

.. figure:: /_static/img/epoch_metrics.png
   :scale: 25 %
   :alt: Epoch metric graphs in tensorboard

Note that we also get the batch metrics here. In fact this is the terminal value of the batch metric, which means that
by default it is an average over the last 50 batches. This can be useful when looking at over-fitting as it gives a more
accurate depiction of the model performance on the training data (the other training metrics are an average over the
whole epoch despite model performance changing throughout).

Source Code
------------------------------------

The source code for these examples is given below:

 :download:`Download Python source code: tensorboard.py </_static/examples/tensorboard.py>`
