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

# Sets global variable notebook when line magic is called with "%torchbearer notebook"
global notebook
notebook = False
from IPython.core.magic import register_line_magic
@register_line_magic
def torchbearer(line):
    if line == 'notebook':
        global notebook
        notebook = True
del torchbearer  # Avoid scope issues

from .cite import cite
from .bases import *
from .state import *

from . import metrics
from . import callbacks
from .trial import *
from . import cv_utils
from . import variational
