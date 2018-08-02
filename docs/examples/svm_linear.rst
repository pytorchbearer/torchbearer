Linear Support Vector Machine (SVM)
===================================

We've seen how to frame a problem as a differentiable program in the `Optimising Functions example <./basic_opt.html>`_.
Now we can take a look a more usable example; a linear Support Vector Machine (SVM). Note that the model and loss used
in this guide are based on the code found `here <https://github.com/kazuto1011/svm-pytorch>`_.

SVM Recap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recall that an SVM tries to find the maximum margin hyperplane which separates the data classes. For a soft margin SVM
where :math:`\textbf{x}` is some data point, we minimize:

:math:`\left[\frac 1 n \sum_{i=1}^n \max\left(0, 1 - y_i(\textbf{w}\cdot \textbf{x}_i - b)\right) \right] + \lambda\lVert \textbf{w} \rVert^2`


We can formulate this as an optimization over our weights :math:`\textbf{w}` and bias :math:`b`, where we minimize the
hinge loss subject to a level 2 weight decay term. The hinge loss for some model outputs :math:`z` is given by:

:math:`\ell(y,z) = \max\left(0, 1 - yz \right)`

Defining the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's put this into code. First we can define our module which will project the data through our weights and offset by
a bias. Note that this is identical to the function of a linear layer.

.. literalinclude:: /_static/examples/svm_linear.py
   :language: python
   :lines: 20-30

Next, we define the hinge loss function:

.. literalinclude:: /_static/examples/svm_linear.py
   :language: python
   :lines: 33-34

Creating Synthetic Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now for some data. :math:`1024` samples should do the trick:

.. literalinclude:: /_static/examples/svm_linear.py
   :language: python
   :lines: 37-40

.. figure:: /_static/img/svm_fit.gif
   :scale: 100 %
   :alt: Convergence of the SVM decision boundary

