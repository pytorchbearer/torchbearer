torchbearer.callbacks
====================================

Base Classes
------------------------------------
..  autoclass:: torchbearer.bases.Callback
        :member-order: bysource
        :members:
        :undoc-members:

..  automodule:: torchbearer.callbacks.callbacks
        :member-order: bysource
        :members:
        :undoc-members:

Imaging
------------------------------------

Main Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
..  automodule:: torchbearer.callbacks.imaging.imaging
        :members:
        :undoc-members:

Deep Inside Convolutional Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torchbearer.callbacks.imaging.inside_cnns
        :members:
        :undoc-members:

Model Checkpointers
------------------------------------

..  automodule:: torchbearer.callbacks.checkpointers
        :members:
        :undoc-members:

Logging
------------------------------------

..  automodule:: torchbearer.callbacks.csv_logger
        :members:
        :undoc-members:

..  automodule:: torchbearer.callbacks.printer
        :members:
        :undoc-members:

Tensorboard, Visdom and Others
------------------------------------

..  automodule:: torchbearer.callbacks.tensor_board
        :members:
        :undoc-members:

..  autoclass:: torchbearer.callbacks.live_loss_plot.LiveLossPlot
        :members:
        :undoc-members:

Early Stopping
------------------------------------

..  automodule:: torchbearer.callbacks.early_stopping
        :members:
        :undoc-members:

..  automodule:: torchbearer.callbacks.terminate_on_nan
        :members:
        :undoc-members:

Gradient Clipping
------------------------------------

..  automodule:: torchbearer.callbacks.gradient_clipping
        :members:
        :undoc-members:

Learning Rate Schedulers
------------------------------------

..  automodule:: torchbearer.callbacks.torch_scheduler
        :members:
        :undoc-members:

Weight Decay
------------------------------------

..  automodule:: torchbearer.callbacks.weight_decay
        :members:
        :undoc-members:

Weight / Bias Initialisation
------------------------------------

..  automodule:: torchbearer.callbacks.init
        :members:
        :undoc-members:

Regularisers
------------------------------------

..  autoclass:: torchbearer.callbacks.cutout.Cutout
        :members:
        :undoc-members:

..  autoclass:: torchbearer.callbacks.cutout.RandomErase
        :members:
        :undoc-members:

..  autoclass:: torchbearer.callbacks.cutout.CutMix
        :members:
        :undoc-members:

..  autoclass:: torchbearer.callbacks.mixup.Mixup
        :members:
        :undoc-members:

..  autoclass:: torchbearer.callbacks.between_class.BCPlus
        :members:
        :undoc-members:

..  autoclass:: torchbearer.callbacks.sample_pairing.SamplePairing
        :members:
        :undoc-members:

..  autoclass:: torchbearer.callbacks.label_smoothing.LabelSmoothingRegularisation
        :members:
        :undoc-members:

Unpack State
------------------------------------

..  autoclass:: torchbearer.callbacks.unpack_state
        :members:
        :undoc-members:


Decorators
------------------------------------

Main
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The main callback decorators simply take a function and bind it to a callback point, returning the result.

.. autofunction:: torchbearer.callbacks.decorators.on_init
.. autofunction:: torchbearer.callbacks.decorators.on_start
.. autofunction:: torchbearer.callbacks.decorators.on_start_epoch
.. autofunction:: torchbearer.callbacks.decorators.on_start_training
.. autofunction:: torchbearer.callbacks.decorators.on_sample
.. autofunction:: torchbearer.callbacks.decorators.on_forward
.. autofunction:: torchbearer.callbacks.decorators.on_criterion
.. autofunction:: torchbearer.callbacks.decorators.on_backward
.. autofunction:: torchbearer.callbacks.decorators.on_step_training
.. autofunction:: torchbearer.callbacks.decorators.on_end_training
.. autofunction:: torchbearer.callbacks.decorators.on_start_validation
.. autofunction:: torchbearer.callbacks.decorators.on_sample_validation
.. autofunction:: torchbearer.callbacks.decorators.on_forward_validation
.. autofunction:: torchbearer.callbacks.decorators.on_criterion_validation
.. autofunction:: torchbearer.callbacks.decorators.on_step_validation
.. autofunction:: torchbearer.callbacks.decorators.on_end_validation
.. autofunction:: torchbearer.callbacks.decorators.on_end_epoch
.. autofunction:: torchbearer.callbacks.decorators.on_checkpoint
.. autofunction:: torchbearer.callbacks.decorators.on_end

Utility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Alongside the base callback decorators that simply bind a function to a callback point, Torchbearer has a number of utility decorators that help simplify callback construction.

..  automodule:: torchbearer.callbacks.decorators
        :noindex: LambdaCallback, bind_to, count_args, on_init, on_start, on_start_epoch, on_start_training, on_sample, on_forward, on_criterion, on_backward, on_step_training, on_step_training, on_end_training, on_start_validation, on_sample_validation, on_forward_validation, on_criterion_validation, on_step_validation, on_end_validation, on_end_epoch, on_checkpoint, on_end
        :exclude-members: LambdaCallback, bind_to, count_args, on_init, on_start, on_start_epoch, on_start_training, on_sample, on_forward, on_criterion, on_backward, on_step_training, on_step_training, on_end_training, on_start_validation, on_sample_validation, on_forward_validation, on_criterion_validation, on_step_validation, on_end_validation, on_end_epoch, on_checkpoint, on_end
        :members:
        :undoc-members:
