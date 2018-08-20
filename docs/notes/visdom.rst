Logging to Visdom
=================================================

In this note we will cover the use of the :class:`TensorBoard callback <.TensorBoard>` to log to visdom.
See the `tensorboard note <../notes/tensorboard.html>`_ for more on the callback in general.

Model Setup
------------------------------------

We'll use the same setup as the `tensorboard note <../notes/tensorboard.html>`_.

.. literalinclude:: /_static/examples/visdom_note.py
   :lines: 9-26

.. literalinclude:: /_static/examples/visdom_note.py
   :lines: 29-55

Logging Epoch and Batch Metrics
------------------------------------
Visdom does not support logging model graphs so we shall start with logging epoch and batch metrics.
The only change we need to make to the tensorboard example is setting :code:`visdom=True` in the :class:`TensorBoard callback <.TensorBoard>` constructor.

.. literalinclude:: /_static/examples/visdom_note.py
   :lines: 60-62

If your visdom server is running then you should see something similar to the figure below:

.. figure:: /_static/img/visdom_main.png
   :scale: 25 %
   :alt: Visdom logging batch and epoch statistics


Visdom Client Parameters
------------------------------------
The visdom client defaults to logging to localhost:8097 in the main environment however this is rather restrictive.
We would like to be able to log to any server on any port and in any environment.
To do this we need to edit the :class:`.VisdomParams` class.

.. code:: python

    class VisdomParams:
        """ ... """
        SERVER = 'http://localhost'
        ENDPOINT = 'events'
        PORT = 8097
        IPV6 = True
        HTTP_PROXY_HOST = None
        HTTP_PROXY_PORT = None
        ENV = 'main'
        SEND = True
        RAISE_EXCEPTIONS = None
        USE_INCOMING_SOCKET = True
        LOG_TO_FILENAME = None

We first import the tensorboard file.

.. literalinclude:: /_static/examples/visdom_note.py
   :lines: 64

We can then edit the visdom client parameters, for example, changing the environment to "Test".

.. literalinclude:: /_static/examples/visdom_note.py
   :lines: 66

Running another fit call, we can see we are now logging to the "Test" environment.

.. figure:: /_static/img/visdom_test.png
   :scale: 25 %
   :alt: Visdom logging to new environment

The only paramenter that the :class:`TensorBoard callback <.TensorBoard>` sets explicity (and cannot be overrided) is the `LOG_TO_FILENAME` parameter.
This is set to the `log_dir` given on the callback init.

Source Code
------------------------------------

The source code for this example is given below:

 :download:`Download Python source code: visdom.py </_static/examples/visdom_note.py>`
