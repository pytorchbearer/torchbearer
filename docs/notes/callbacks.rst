The Callback API
====================================

This note gives a very brief, high level overview of the callback API. For a much more detailed discussion and guides on
specific callbacks see the `example library <https://github.com/pytorchbearer/torchbearer#examples>`_.

Aims
------------------------------------

The aim of callbacks is to enable the user to dynamically change the behaviour of the core training loop of a trial. To
do this, callbacks are given state which is a dictionary containing all of the variables currently in use by a trial.
Furthermore, callbacks can be made persistent by implementing the `state_dict` and `load_state_dict` methods, these will
be automatically handled by :meth:`.Trial.state_dict` and :meth:`.Trial.load_state_dict`.

A key property of torchbearers construction is that the user is permitted to do more or less anything without triggering
errors or warnings. That is, we assume all users are super users and that all actions are taken deliberately. A
consequence of this approach is that some level of debugging skill can be required to get complex things to work
correctly. However, the benefit of this approach is that torchbearer is a uniquely flexible library that never tries to
stop you doing something you want to do. To understand further how the training process can be changed have a look at
the `callbacks example <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/callbacks.ipynb>`_
and the `custom data loaders example <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/custom_loaders.ipynb>`_

Fluent
------------------------------------

Fluent interfaces are used in various places in torchbearer. In particular, many of our built-in callbacks employ method
chaining and naturally readable method names in order to promote more readable code and a more usable API. A typical
example would be the base :class:`.ImagingCallback` class. To see this in action have a look at the
`imaging example <https://nbviewer.jupyter.org/github/pytorchbearer/torchbearer/blob/master/docs/_static/notebooks/imaging.ipynb>`_.
For more information on fluent take a look at the `original blog post <https://martinfowler.com/bliki/FluentInterface.html>`_.