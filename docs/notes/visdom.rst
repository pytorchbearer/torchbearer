Using Visdom Logging in the Tensorboard Callbacks
====================================

In this note we will cover the use of the :class:`TensorBoard callback <.TensorBoard>` to log to visdom, see the
`tensorboard note <../notes/tensorboard.html>`_ for setup and more on the callback in general.

Model Setup
------------------------------------

We'll use the same setup as the `tensorboard note <../notes/tensorboard.html>`_.

.. literalinclude:: /_static/examples/tensorboard.py
   :lines: 9-26

.. literalinclude:: /_static/examples/tensorboard.py
   :lines: 29-55

Logging Batch Metrics
------------------------------------
Visdom does not support logging model graphs so we shall start with logging batch metrics.