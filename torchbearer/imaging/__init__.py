"""
Main Classes
------------------------------------
..  automodule:: torchbearer.callbacks.imaging.imaging
        :members:
        :undoc-members:

Deep Inside Convolutional Networks
------------------------------------
.. automodule:: torchbearer.callbacks.imaging.inside_cnns
        :members:
        :undoc-members:
"""
from .imaging import *
from .inside_cnns import ClassAppearanceModel, RANDOM
from .loss import *
from .images import *
from . import transforms
from .ascent import *


def check_vision():
    from distutils.version import LooseVersion
    try:
        import torchvision
        return LooseVersion(torchvision.__version__) >= LooseVersion("0.3.0")
    except ImportError:
        return False


if check_vision():
    from . import models
else:
    class _Models(object):
        def __getattr__(self, _):
            raise ImportError('Cannot import models - requires torchvision >= 0.3.0')
    models = _Models()
