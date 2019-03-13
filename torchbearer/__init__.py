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
from . import magics
from .cite import cite
from .bases import *
from .state import *

from . import metrics
from . import callbacks
from .trial import *
from . import cv_utils
from . import variational
