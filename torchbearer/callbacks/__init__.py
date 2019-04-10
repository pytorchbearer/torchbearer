"""
Base Classes
------------------------------------

..  automodule:: torchbearer.callbacks.callbacks
        :members:
        :undoc-members:

Imaging
------------------------------------

..  automodule:: torchbearer.callbacks.imaging
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


Learning Rate Finders
------------------------------------

..  automodule:: torchbearer.callbacks.lr_finder
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

Decorators
------------------------------------

..  automodule:: torchbearer.callbacks.decorators
        :members:
        :undoc-members:
"""

from torchbearer import Callback
from .callbacks import *
from .checkpointers import *
from .csv_logger import *
from .early_stopping import *
from .gradient_clipping import *
from .printer import ConsolePrinter, Tqdm
from .tensor_board import TensorBoard, TensorBoardImages, TensorBoardProjector, TensorBoardText
from .terminate_on_nan import *
from .torch_scheduler import *
from .weight_decay import *
from .aggregate_predictions import *
from .decorators import *
from .live_loss_plot import LiveLossPlot
from . import init
from . import imaging
