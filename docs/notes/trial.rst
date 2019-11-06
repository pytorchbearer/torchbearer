The Trial Class
====================================

The core class in torchbearer is :class:`.Trial`. This class contains the :meth:`.Trial.run` method which performs the
main train-eval loop. In this note we'll go into detail regarding how to create and use a trial. We'll start with
instantiation before looking at loading data (or running without data) and finally covering means of controlling
verbosity. To get an understanding of actually running a trial and using metrics / callbacks, have a look at our
`example library <https://github.com/pytorchbearer/torchbearer#examples>`_.

Instantiation
------------------------------------
If desired, we can instantiate a trial with no arguments by passing `None` as the `model` argument. However, assuming we
have a model, we can simply write:

.. code-block:: python

    from torchbearer import Trial
    trial = Trial(model)

If we would like our trial also perform optimization of some criterion we can use:

.. code-block:: python

    from torchbearer import Trial
    trial = Trial(model, optimizer, criterion)

Criterions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Criterions can be given to a trial either as standard `torch.nn` criterions (i.e. functions of y_true, y_pred) or as
functions of state. State is passed around a lot in torchbearer and contains (mutably) all of the different variables
that are relevant to the fitting process at that point in time. Underneath it's just a dictionary but should be accessed
with :class:`.StateKey` objects. The list of built-in state keys can be found `here <../code/main.html#key-list>`_. In
the case of criterions, this

.. code-block:: python

    from torchbearer import Trial

    def my_criterion(y_pred, y_true):
        return (y_pred - y_true).abs()

    trial = Trial(model, optimizer, my_criterion)

is equivalent to

.. code-block:: python

    import torchbearer
    from torchbearer import Trial

    def my_criterion(state):
        return (state[torchbearer.PREDICTION] - state[torchbearer.TARGET]).abs()

    trial = Trial(model, optimizer, my_criterion)


Loading Data
------------------------------------

To load data into a trial we can use generators (such as a `torch.utils.data.DataLoader`) or tensors. We can further
use different methods for loading train, val and test data or load all at once. Finally, we can let torchbearer decide
how many steps per epoch to perform or tell it explicitly. All of these methods mutate the underlying trial and are
chainable.

Generators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To populate the trial with a train generator we can use the :meth:`.Trial.with_train_generator` method. Equivalent
methods :meth:`.Trial.with_val_generator` and :meth:`.Trial.with_test_generator` exist for validation and test data
respectively. To use this method we write

.. code-block:: python

    from torchbearer import Trial

    trial = Trial(model).with_train_generator(train_loader)


For simplicity, we can also load several data sets in one call using :meth:`.Trial.with_generators` like this

.. code-block:: python

    from torchbearer import Trial

    trial = Trial(model).with_generators(train_loader, val_loader, test_loader)


To control the number of steps, we can either pass an integer argument `steps` to the `with_XXX_generator` methods or
pass `train_steps`, `val_steps` and `test_steps` individually to :meth:`.Trial.with_generators`. Finally, we can use:
:meth:`.Trial.for_train_steps`, :meth:`.Trial.for_val_steps`, :meth:`.Trial.for_test_steps`, :meth:`.Trial.for_steps`.
That is, the following are all equivalent

.. code-block:: python

    trial = Trial(model).with_train_generator(train_loader, steps=10)
    trial = Trial(model).with_generators(train_loader, train_steps=10)
    trial = Trial(model).with_train_generator(train_loader).for_train_steps(10)
    etc.

A final option is to tell the trial to run for infinitely many training steps (until stopped) for which we can use
:meth:`.Trial.with_inf_train_loader`. For example

.. code-block:: python

    trial = Trial(model).with_train_generator(train_loader).with_inf_loader()


For more info on data loaders see the
`custom data loaders example <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/custom_loaders.ipynb>`_

Tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we want to load tensors instead we can use the `with_XXX_data` methods or the :meth:`.Trial.with_data` method in much
the same way as before. There are some additional arguments to control batch size, shuffle and number of workers. Here
are some examples:

.. code-block:: python

    # Shuffled training data
    trial = Trial(model).with_train_data(x, y, shuffle=True, batch_size=128)

    # Test data (no targets)
    trial = Trial(model).with_test_data(x, batch_size=128)

    # with_data
    trial = Trial(model).with_data(x_train, y_train, x_val, y_val, x_test, shuffle=True, batch_size=128)

To change the number of steps we can use the same `steps` arguments or `for_steps` methods as before.

Running Without Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we want to run an optimisation or similar which does not require data, we simply call the `for_steps` methods without
calling any `with_generator` / `with_data` methods. For example, to run for 100 train steps per cycle without any data,
we use:

.. code-block:: python

    trial = Trial(model).for_train_steps(100)

In this case, the model will be given `None` as input at each step.

Controlling Verbosity
------------------------------------

The verbosity of a trial can be controlled in two ways. First, a global verbosity is set in the init. Second each of the
`run` / `evaluate` / `predict` methods can take a local verbosity argument which gets priority. If `verbose=2`, the
:class:`.Tqdm` callback will be loaded with `on_epoch=False` so that a new progress bar is created for each epoch. If
`verbose=1`, the :class:`.Tqdm` callback will be loaded with `on_epoch=True` so that only one progress bar is created.
If `verbose=0`, no :class:`.Tqdm` callback will be loaded so that the trial produces no output. The default behaviour
is `verbose=2`. For example, to suppress output we can use:

.. code-block:: python

    trial = Trial(model)
    trial.run(10, verbose=0)
