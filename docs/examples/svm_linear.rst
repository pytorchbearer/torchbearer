Linear Support Vector Machine (SVM)
===================================

We've seen how to frame a problem as a differentiable program in the `Optimising Functions example <./basic_opt.html>`_.
Now we can take a look a more usable example; a linear Support Vector Machine (SVM). Note that the model and loss used
in this guide are based on the code found `here <https://github.com/kazuto1011/svm-pytorch>`_.

SVM Recap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recall that an SVM tries to find the maximum margin hyperplane which separates the data classes. For a soft margin SVM
where :math:`\textbf{x}` is our data, we minimize:

:math:`\left[\frac 1 n \sum_{i=1}^n \max\left(0, 1 - y_i(\textbf{w}\cdot \textbf{x}_i - b)\right) \right] + \lambda\lVert \textbf{w} \rVert^2`


We can formulate this as an optimization over our weights :math:`\textbf{w}` and bias :math:`b`, where we minimize the
hinge loss subject to a level 2 weight decay term. The hinge loss for some model outputs
:math:`z = \textbf{w}\textbf{x} + b` with targets :math:`y` is given by:

:math:`\ell(y,z) = \max\left(0, 1 - yz \right)`

Defining the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's put this into code. First we can define our module which will project the data through our weights and offset by
a bias. Note that this is identical to the function of a linear layer.

.. literalinclude:: /_static/examples/svm_linear.py
   :language: python
   :lines: 17-27

Next, we define the hinge loss function:

.. literalinclude:: /_static/examples/svm_linear.py
   :language: python
   :lines: 30-31

Creating Synthetic Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now for some data, 1024 samples should do the trick. We normalise here so that our random init is in the same space as
the data:

.. literalinclude:: /_static/examples/svm_linear.py
   :language: python
   :lines: 34-37

Subgradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since we don't know that our data is linearly separable, we would like to use a soft-margin SVM. That is, an SVM for
which the data does not all have to be outside of the margin. This takes the form of a weight decay term,
:math:`\lambda\lVert \textbf{w} \rVert^2` in the above equation. This term is called weight decay because the gradient
corresponds to subtracting some amount (:math:`2\lambda\textbf{w}`) from our weights at each step. With torchbearer we
can use the :class:`.L2WeightDecay` callback to do this. This whole process is known as subgradient descent because we
only use a mini-batch (of size 32 in our example) at each step to approximate the gradient over all of the data. This is
proven to converge to the minimum for convex functions such as our SVM. At this point we are ready to create and train
our model:

.. literalinclude:: /_static/examples/svm_linear.py
   :language: python
   :lines: 91-98

Visualizing the Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You might have noticed some strange things in the :meth:`.Trial` callbacks list. Specifically, we use the
:class:`.ExponentialLR` callback to anneal the convergence a little and we have a couple of other callbacks:
:code:`scatter` and :code:`draw_margin`. These callbacks produce the following live visualisation (note, doesn't work in
PyCharm, best run from terminal):

.. figure:: /_static/img/svm_fit.gif
   :scale: 100 %
   :alt: Convergence of the SVM decision boundary

The code for the visualisation (using `pyplot <https://matplotlib.org/api/pyplot_api.html>`_) is a bit ugly but we'll
try to explain it to some degree. First, we need a mesh grid :code:`xy` over the range of our data:

.. literalinclude:: /_static/examples/svm_linear.py
   :language: python
   :lines: 40-44

Next, we have the scatter callback. This happens once at the start of our fit call and draws the figure with a scatter
plot of our data:

.. literalinclude:: /_static/examples/svm_linear.py
   :language: python
   :lines: 60-64

Now things get a little strange. We start by evaluating our model over the mesh grid from earlier:

.. literalinclude:: /_static/examples/svm_linear.py
   :language: python
   :lines: 67-71

For our outputs :math:`z \in \textbf{Z}`, we can make some observations about the decision boundary. First, that we are
outside the margin if :math:`z \lt -1` or :math:`z \gt 1`. Conversely, we are inside the margine where :math:`z \gt -1`
or :math:`z \lt 1`. This gives us some rules for colouring, which we use here:

.. literalinclude:: /_static/examples/svm_linear.py
   :language: python
   :lines: 73-77

So far it's been relatively straight forward. The next bit is a bit of a hack to get the update of the contour plot
working. If a reference to the plot is already in state we just remove the old one and add a new one, otherwise we add
it and show the plot. Finally, we call :code:`mypause` to trigger an update. You could just use :code:`plt.pause`,
however, it grabs the mouse focus each time it is called which can be annoying. Instead, :code:`mypause` is taken from
`stackoverflow <https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7>`_.

.. literalinclude:: /_static/examples/svm_linear.py
   :language: python
   :lines: 79-88

Final Comments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So, there you have it, a fun differentiable programming example with a live visualisation in under 100 lines of code
with torchbearer. It's easy to see how this could become more useful, perhaps finding a way to use the kernel trick with
the standard form of an SVM (essentially an RBF network). You could also attempt to write some code that saves the gif
from earlier. We had some but it was beyond a hack, can you do better?

Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source code for the example is given below:

 :download:`Download Python source code: svm_linear.py </_static/examples/svm_linear.py>`
