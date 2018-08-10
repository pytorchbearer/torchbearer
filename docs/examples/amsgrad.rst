Breaking ADAM
====================================

In case you haven't heard, one of the top papers at `ICLR 2018 <https://iclr.cc/Conferences/2018>`_ (pronounced:
eye-clear, who knew?) was `On the Convergence of Adam and Beyond <https://openreview.net/forum?id=ryQu7f-RZ>`_. In the
paper, the authors determine a flaw in the convergence proof of the ubiquitous ADAM optimizer. They also give an example
of a simple function for which ADAM does not converge to the correct solution. We've seen how torchbearer can be used
for `simple function optimization <./basic_opt.html>`_ before and we can do something similar to reproduce the results
from the paper.

Online Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Online learning basically just means learning from one example at a time, in sequence. The function given in the paper
is defined as follows:

:math:`f_t(x) = \begin{cases}1010x, & \text{for } t \; \texttt{mod} \; 101 = 1 \\ -10x, & \text{otherwise}\end{cases}`

We can then write this as a PyTorch model whose forward is a function of its parameters with the following:

.. literalinclude:: /_static/examples/amsgrad.py
   :language: python
   :lines: 10-25

We now define a loss (simply return the model output) and a metric which returns the value of our parameter :math:`x`:

.. literalinclude:: /_static/examples/amsgrad.py
   :language: python
   :lines: 46-56

In the paper, :math:`x` can only hold values in :math:`[-1, 1]`. We don't strictly need to do anything but we can write
a callback that greedily updates :math:`x` if it is outside of its range as follows:

.. literalinclude:: /_static/examples/amsgrad.py
   :language: python
   :lines: 59-64

Finally, we can train this model twice; once with ADAM and once with AMSGrad (included in PyTorch) with just a few
lines:

.. literalinclude:: /_static/examples/amsgrad.py
   :language: python
   :lines: 67-77

Note that we have logged to TensorBoard here and after completion, running :code:`tensorboard --logdir logs` and
navigating to `localhost:6006 <http://localhost:6006>`_, we can see a graph like the one in Figure 1 from the paper,
where the top line is with ADAM and the bottom with AMSGrad:

.. figure:: /_static/img/ams_grad_online.png
   :scale: 50 %
   :alt: ADAM failure case - online

Stochastic Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To simulate a stochastic setting, the authors use a slight variant of the function, which changes with some probability:

:math:`f_t(x) = \begin{cases}1010x, & \text{with probability } 0.01 \\ -10x, & \text{otherwise}\end{cases}`

We can again formulate this as a PyToch model:

.. literalinclude:: /_static/examples/amsgrad.py
   :language: python
   :lines: 28-43

Using the loss, callback and metric from our previous example, we can train with the following:

.. literalinclude:: /_static/examples/amsgrad.py
   :language: python
   :lines: 79-87

After execution has finished, again running :code:`tensorboard --logdir logs` and navigating to
`localhost:6006 <http://localhost:6006>`_, we see another graph similar to that of the stochastic setting in Figure 1 of
the paper, where the top line is with ADAM and the bottom with AMSGrad:

.. figure:: /_static/img/ams_grad_stochastic.png
   :scale: 50 %
   :alt: ADAM failure case - stochastic

Conclusions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So, whatever your thoughts on the AMSGrad optimizer in practice, it's probably the sign of a good paper that you can
re-implement the example and get very similar results without having to try too hard and (thanks to torchbearer) only
writing a small amount of code. The paper includes some more complex, 'real-world' examples, can you re-implement those
too?

Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for this example can be downloaded below:

 :download:`Download Python source code: amsgrad.py </_static/examples/amsgrad.py>`