torchbearer
====================================

Trial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  autoclass:: torchbearer.Trial
        :member-order: bysource
        :members:
        :undoc-members:


Batch Loaders
--------------------
..  autofunction:: torchbearer.trial.load_batch_infinite
..  autofunction:: torchbearer.trial.load_batch_none
..  autofunction:: torchbearer.trial.load_batch_predict
..  autofunction:: torchbearer.trial.load_batch_standard

Misc
--------------------
..  autofunction:: torchbearer.trial.deep_to
..  autofunction:: torchbearer.trial.update_device_and_dtype



State
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The state is central in torchbearer, storing all of the relevant intermediate values that may be changed or replaced
during model fitting. This module defines classes for interacting with state and all of the built in state keys used
throughout torchbearer. The :func:`state_key` function can be used to create custom state keys for use in callbacks or
metrics.

Example: ::

    >>> from torchbearer import state_key
    >>> MY_KEY = state_key('my_test_key')

State
--------------------
..  automodule:: torchbearer.state
        :members: State, StateKey, state_key
        :undoc-members:

Key List
--------------------
..  automodule:: torchbearer.state
        :noindex: State, StateKey, state_key
        :exclude-members: State, StateKey, state_key
        :members:
        :undoc-members:

Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  automodule:: torchbearer.cv_utils
        :members:
        :undoc-members:

.. autofunction:: torchbearer.bases.base_closure

