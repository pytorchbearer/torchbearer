"""
Trial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  automodule:: torchbearer.trial
        :members:
        :undoc-members:

Model (Deprecated)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  automodule:: torchbearer.torchbearer
        :members:
        :undoc-members:

State
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  automodule:: torchbearer.state
        :members:
        :undoc-members:

Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  automodule:: torchbearer.cv_utils
        :members:
        :undoc-members:
"""

from .version import __version__

# Hack: Tensorboard and metrics need Y_PRED and Y_TRUE so define them here as a string
Y_PRED = 'y_pred'
Y_TRUE = 'y_true'

from . import metrics
from . import callbacks
from .state import *
from .trial import *
from .torchbearer import *
from . import cv_utils
