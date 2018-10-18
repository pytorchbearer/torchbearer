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

Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

..  automodule:: torchbearer.state
        :members:
        :undoc-members:

..  automodule:: torchbearer.cv_utils
        :members:
        :undoc-members:
"""

from .version import __version__

# Hack: Tensorboard needs Y_PRED so define it here as a string
Y_PRED = 'y_pred'

from . import metrics
from . import callbacks
from .state import *
from .trial import *
from .torchbearer import *
from . import cv_utils
